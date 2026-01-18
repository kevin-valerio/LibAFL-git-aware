use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use crate::Error;

pub(crate) const GIT_RECENCY_MAPPING_ENV: &str = "LIBAFL_GIT_RECENCY_MAPPING_PATH";

const SIDECAR_MAGIC: &[u8; 8] = b"LAFLGIT1";
pub(crate) const SIDECAR_EXT: &str = "libafl_git_recency";

#[derive(Debug, Clone)]
struct SidecarEntry {
    file: Option<String>,
    line: u32,
}

fn sidecar_path_for_object(obj: &Path) -> PathBuf {
    let mut s: OsString = obj.as_os_str().to_os_string();
    s.push(".");
    s.push(SIDECAR_EXT);
    PathBuf::from(s)
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

fn write_u64_le(out: &mut impl Write, v: u64) -> Result<(), Error> {
    out.write_all(&v.to_le_bytes()).map_err(Error::Io)
}

fn parse_sidecar(bytes: &[u8]) -> Result<Vec<SidecarEntry>, Error> {
    if bytes.len() < 16 {
        return Err(Error::Unknown(
            "git recency sidecar too small to be valid".to_string(),
        ));
    }
    if &bytes[0..8] != SIDECAR_MAGIC {
        return Err(Error::Unknown(
            "git recency sidecar magic mismatch".to_string(),
        ));
    }

    let len = read_u64_le(&bytes[8..16]);
    let len = usize::try_from(len)
        .map_err(|_| Error::Unknown("git recency sidecar length does not fit usize".to_string()))?;

    let mut entries = Vec::with_capacity(len);
    let mut offset = 16usize;
    for _ in 0..len {
        if offset + 8 > bytes.len() {
            return Err(Error::Unknown(
                "git recency sidecar truncated while reading entry header".to_string(),
            ));
        }
        let line = read_u32_le(&bytes[offset..offset + 4]);
        let path_len = read_u32_le(&bytes[offset + 4..offset + 8]) as usize;
        offset += 8;

        if offset + path_len > bytes.len() {
            return Err(Error::Unknown(
                "git recency sidecar truncated while reading path".to_string(),
            ));
        }

        let file = if line == 0 || path_len == 0 {
            None
        } else {
            let path_bytes = &bytes[offset..offset + path_len];
            offset += path_len;
            Some(String::from_utf8(path_bytes.to_vec()).map_err(|e| {
                Error::Unknown(format!("git recency sidecar contains non-utf8 path: {e}"))
            })?)
        };
        if file.is_none() {
            // If unknown, ignore any path bytes already accounted for.
            offset += path_len;
        }

        entries.push(SidecarEntry { file, line });
    }

    if offset != bytes.len() {
        return Err(Error::Unknown(
            "git recency sidecar has trailing bytes".to_string(),
        ));
    }

    Ok(entries)
}

fn git(repo_root: &Path, args: &[&str]) -> Result<Output, Error> {
    Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .map_err(Error::Io)
}

fn repo_root(cwd: &Path) -> Result<PathBuf, Error> {
    let out = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(Error::Io)?;

    if !out.status.success() {
        return Err(Error::Unknown(format!(
            "git rev-parse --show-toplevel failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let root = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if root.is_empty() {
        return Err(Error::Unknown(
            "git rev-parse --show-toplevel returned empty output".to_string(),
        ));
    }
    Ok(PathBuf::from(root))
}

fn head_time_epoch_seconds(repo_root: &Path) -> Result<u64, Error> {
    let out = git(repo_root, &["show", "-s", "--format=%ct", "HEAD"])?;
    if !out.status.success() {
        return Err(Error::Unknown(format!(
            "git show failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    s.parse::<u64>()
        .map_err(|e| Error::Unknown(format!("failed to parse HEAD time '{s}': {e}")))
}

fn is_header_line(line: &str) -> bool {
    let mut it = line.split_whitespace();
    let Some(hash) = it.next() else {
        return false;
    };
    let Some(orig_line) = it.next() else {
        return false;
    };
    let Some(final_line) = it.next() else {
        return false;
    };

    if !hash.chars().all(|c| c == '^' || c.is_ascii_hexdigit()) {
        return false;
    }
    if orig_line.parse::<u32>().is_err() {
        return false;
    }
    if final_line.parse::<u32>().is_err() {
        return false;
    }
    true
}

fn blame_times_for_lines(
    repo_root: &Path,
    file_rel: &str,
    needed_lines: &HashSet<u32>,
) -> Result<HashMap<u32, u64>, Error> {
    let (min_line, max_line) = needed_lines
        .iter()
        .fold((u32::MAX, 0u32), |acc, &v| (acc.0.min(v), acc.1.max(v)));
    if min_line == u32::MAX || max_line == 0 {
        return Ok(HashMap::new());
    }

    let range = format!("{min_line},{max_line}");
    let out = git(
        repo_root,
        &["blame", "--line-porcelain", "-L", &range, "--", file_rel],
    )?;

    if !out.status.success() {
        // Treat failures as "unknown/old", per plan.
        return Ok(HashMap::new());
    }

    let text = String::from_utf8_lossy(&out.stdout);
    let mut res: HashMap<u32, u64> = HashMap::new();

    let mut current_final_line: Option<u32> = None;
    let mut current_committer_time: Option<u64> = None;

    for line in text.lines() {
        if current_final_line.is_none() && is_header_line(line) {
            let mut it = line.split_whitespace();
            let _hash = it.next().unwrap();
            let _orig = it.next().unwrap();
            let final_line = it.next().unwrap();
            current_final_line = final_line.parse::<u32>().ok();
            current_committer_time = None;
            continue;
        }

        if let Some(rest) = line.strip_prefix("committer-time ") {
            current_committer_time = rest.trim().parse::<u64>().ok();
            continue;
        }

        if line.starts_with('\t') {
            if let (Some(final_line), Some(time)) = (current_final_line, current_committer_time)
                && needed_lines.contains(&final_line)
            {
                res.insert(final_line, time);
            }
            current_final_line = None;
            current_committer_time = None;
        }
    }

    Ok(res)
}

pub(crate) fn generate_git_recency_mapping(
    mapping_out: &Path,
    object_files: &[PathBuf],
    cwd: &Path,
) -> Result<(), Error> {
    let repo_root = repo_root(cwd)?;
    let repo_root = fs::canonicalize(&repo_root).map_err(Error::Io)?;

    let head_time = head_time_epoch_seconds(&repo_root)?;

    let mut resolved: Vec<Option<(String, u32)>> = Vec::new();
    for obj in object_files {
        let sidecar_path = sidecar_path_for_object(obj);
        let sidecar_bytes = fs::read(&sidecar_path).map_err(|e| {
            Error::Unknown(format!(
                "missing git recency sidecar for object {}: {e}",
                obj.display()
            ))
        })?;

        let sidecar_entries = parse_sidecar(&sidecar_bytes)?;
        for entry in sidecar_entries {
            let Some(file) = entry.file else {
                resolved.push(None);
                continue;
            };

            if entry.line == 0 {
                resolved.push(None);
                continue;
            }

            let p = PathBuf::from(file);
            let p = if p.is_absolute() { p } else { cwd.join(p) };

            let Ok(p) = fs::canonicalize(&p) else {
                resolved.push(None);
                continue;
            };

            if !p.starts_with(&repo_root) {
                resolved.push(None);
                continue;
            }

            let rel = p.strip_prefix(&repo_root).unwrap();
            let rel = rel.to_string_lossy().replace('\\', "/");
            resolved.push(Some((rel, entry.line)));
        }
    }

    let mut needed_by_file: HashMap<String, HashSet<u32>> = HashMap::new();
    for item in &resolved {
        if let Some((file, line)) = item {
            needed_by_file
                .entry(file.clone())
                .or_default()
                .insert(*line);
        }
    }

    let mut times_by_file: HashMap<String, HashMap<u32, u64>> = HashMap::new();
    for (file, needed_lines) in &needed_by_file {
        let times = blame_times_for_lines(&repo_root, file, needed_lines)?;
        times_by_file.insert(file.clone(), times);
    }

    let mut timestamps: Vec<u64> = Vec::with_capacity(resolved.len());
    for item in resolved {
        let Some((file, line)) = item else {
            timestamps.push(0);
            continue;
        };

        let t = times_by_file
            .get(&file)
            .and_then(|m| m.get(&line))
            .copied()
            .unwrap_or(0);
        timestamps.push(t);
    }

    let mut out = fs::File::create(mapping_out).map_err(Error::Io)?;
    write_u64_le(&mut out, head_time)?;
    write_u64_le(&mut out, timestamps.len() as u64)?;
    for t in timestamps {
        write_u64_le(&mut out, t)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SIDECAR_MAGIC, parse_sidecar};

    #[test]
    fn test_parse_sidecar_empty() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(SIDECAR_MAGIC);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let entries = parse_sidecar(&bytes).unwrap();
        assert!(entries.is_empty());
    }
}
