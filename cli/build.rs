use std::{env, fs, path::Path, process::Command};

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&manifest).parent().unwrap();
    for path in ["Cargo.toml", "Cargo.lock", "src", "cli/src"] {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }
    // Resolve Git paths so this also works in linked worktrees.
    for name in ["HEAD", "packed-refs"] {
        if let Some(path) = git(root, &["rev-parse", "--git-path", name]) {
            println!("cargo:rerun-if-changed={}", root.join(path).display());
        }
    }
    if let Some(reference) = git(root, &["symbolic-ref", "-q", "HEAD"])
        && let Some(path) = git(root, &["rev-parse", "--git-path", &reference])
    {
        println!("cargo:rerun-if-changed={}", root.join(path).display());
    }

    let lock: toml::Value =
        toml::from_str(&fs::read_to_string(root.join("Cargo.lock")).unwrap()).unwrap();
    let packages = lock["package"].as_array().unwrap();
    let cli = packages
        .iter()
        .find(|p| p["name"].as_str() == Some("maroontree-cli"))
        .unwrap();
    let dependency = |name: &str| {
        let id = cli["dependencies"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .find(|id| id.split_whitespace().next() == Some(name))
            .unwrap_or_else(|| panic!("missing CLI dependency {name}"));
        let mut parts = id.split_whitespace();
        parts.next();
        let version = parts.next();
        let source = parts.next().map(|s| s.trim_matches(['(', ')']));
        let package = packages
            .iter()
            .find(|p| {
                p["name"].as_str() == Some(name)
                    && version.is_none_or(|version| p["version"].as_str() == Some(version))
                    && source.is_none_or(|source| {
                        p.get("source")
                            .and_then(|v| v.as_str())
                            .is_some_and(|actual| {
                                actual == source || actual.split('#').next() == Some(source)
                            })
                    })
            })
            .unwrap();
        let version = package["version"].as_str().unwrap();
        match package.get("source").and_then(|v| v.as_str()) {
            Some(source) if source.starts_with("git+") => format!("{name} {version} ({source})"),
            Some("registry+https://github.com/rust-lang/crates.io-index") => {
                format!("{name} {version} (crates.io)")
            }
            Some(source) => format!("{name} {version} ({source})"),
            None => format!("{name} {version} (local source)"),
        }
    };
    let revision = git(root, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(root, &["diff", "--name-only", "HEAD"]).is_some_and(|s| !s.is_empty());
    let revision = format!("{revision}{}", if dirty { "-dirty" } else { "" });
    let own = format!("{} (revision {revision})", dependency("maroontree"));
    let mut generated = format!(
        "pub const CLI: &str = {:?};\n",
        format!(
            "maroontree-cli {} (revision {revision})",
            env::var("CARGO_PKG_VERSION").unwrap()
        )
    );
    for (symbol, description) in [
        ("AV1", format!("av1: {own}")),
        ("AV2", format!("av2: {own}")),
        ("JXL", format!("jxl: {}", dependency("jixel"))),
        (
            "HEVC",
            if env::var_os("CARGO_FEATURE_HEIC").is_some() {
                format!("hevc: {}", dependency("hpvca"))
            } else {
                "hevc: disabled at build time".into()
            },
        ),
        (
            "VVC",
            if env::var_os("CARGO_FEATURE_VVC").is_some() {
                format!("vvc: {}", dependency("garnetash"))
            } else {
                "vvc: disabled at build time".into()
            },
        ),
    ] {
        generated.push_str(&format!("pub const {symbol}: &str = {description:?};\n"));
    }
    fs::write(
        Path::new(&env::var("OUT_DIR").unwrap()).join("versions.rs"),
        generated,
    )
    .unwrap();
}
