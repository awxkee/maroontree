use crate::{
    Args, Chroma, Depth, Encoder, EncodingEffort, Qmatrix, basic_concurrency,
    overrides::OverrideGroup, version,
};
use clap::{Arg, ArgAction, ArgMatches, Command, error::ErrorKind, value_parser};
use std::{ffi::OsString, path::PathBuf};

fn command() -> Command {
    let mut cmd = Command::new("maroontree-cli")
        .about("Encode images as AVIF (AV1/AV2), JPEG XL, HEIC, or VVC HEIF")
        .version(env!("CARGO_PKG_VERSION"))
        .long_version(version::all().trim_start_matches("maroontree-cli ").to_owned())
        .arg_required_else_help(true)
        .after_help("Examples:\n  maroontree-cli input.png\n  maroontree-cli -e jxl -O 'jxl:distance=3.1,speed=medium,progressive=true' input.png\n  maroontree-cli -O 'av1:adaptive-quant=false,cdef=true,qm=7' input.png output.avif\n\nGroup settings with commas and one encoder prefix per -O; repeated -O flags also work.\nOverrides are applied after common options, in order; the last value wins.\nUse -e av1|jxl|heic --list-overrides for supported settings.\nICC profiles and EXIF are preserved by default. Use --no-icc/--no-exif to strip them.")
        .arg(Arg::new("input").value_name("INPUT").value_parser(value_parser!(PathBuf))
            .required_unless_present_any(["encoder-version", "list-overrides"])
            .help("Source image (PNG, JPEG, WebP, TIFF, AVIF, JXL, HEIC, ...)"))
        .arg(Arg::new("output").value_name("OUTPUT").value_parser(value_parser!(PathBuf))
            .help("Destination; defaults to the input path with the encoder's extension"))
        .arg(Arg::new("encoder").short('e').long("encoder").value_name("ENCODER")
            .value_parser(["av1", "av2", "hevc", "heic", "heif", "jxl", "jpegxl", "jpeg-xl", "vvc"])
            .default_value("av1").help("Encoder backend"))
        .arg(Arg::new("quality").short('q').long("quality").value_name("1-100")
            .value_parser(value_parser!(u8).range(1..=100))
            .help("Quality; defaults: AV1/HEVC 80, AV2/VVC 60, JXL 70"))
        .arg(Arg::new("chroma").short('c').long("chroma").value_name("CHROMA")
            .value_parser(["444", "422", "420"]).help("Chroma subsampling [default: 422; lossless: 444]"))
        .arg(Arg::new("depth").short('d').long("depth").value_name("DEPTH")
            .value_parser(["8", "10", "12", "16"]).help("Output bit depth; 16 is JXL-only [default: auto from source]"))
        .arg(Arg::new("threads").short('t').long("threads").value_name("N")
            .value_parser(value_parser!(usize)).default_value("0").help("Worker threads; 0 = all available cores"))
        .arg(Arg::new("speed").short('s').long("speed").value_name("SPEED")
            .value_parser(["extraslow", "slow", "medium", "fast"]).default_value("slow").help("Encoding effort"))
        .arg(Arg::new("qm").long("qm").value_name("auto|0-15")
            .value_parser(parse_qmatrix).help("Enable AV1 quantization matrices"))
        .arg(Arg::new("override").short('O').long("override").value_name("ENCODER:KEY=VALUE[,KEY=VALUE...]")
            .action(ArgAction::Append).value_parser(value_parser!(OverrideGroup))
            .help("Override AV1, JXL, or HEIC settings; comma-separated groups and repeated flags are supported"));
    for (name, short, help) in [
        (
            "lossless",
            None,
            "Encode losslessly (444 required for AV1/AV2/HEVC/VVC)",
        ),
        ("no-alpha", None, "Discard alpha channel"),
        ("no-screen-content", None, "Skip AV1 palette search"),
        ("no-intrabc", None, "Disable lossy AV1 IntraBC"),
        ("cdef", None, "Enable AV1 CDEF"),
        ("wiener", None, "Enable AV1 luma Wiener loop restoration"),
        ("no-exif", None, "Strip EXIF metadata"),
        (
            "no-icc",
            None,
            "Strip ICC profile without converting pixels",
        ),
        (
            "apply-icc",
            None,
            "Convert pixels to sRGB using ICC, then strip the profile",
        ),
        ("no-cdf-update", None, "Freeze AV1 entropy CDFs"),
        (
            "verbose",
            Some('v'),
            "Print encoder version, options, timing, and file statistics",
        ),
        (
            "encoder-version",
            None,
            "Print the selected encoder's version/revision and exit",
        ),
        (
            "list-overrides",
            None,
            "List settings for the selected encoder and exit",
        ),
    ] {
        let mut arg = Arg::new(name)
            .long(name)
            .action(ArgAction::SetTrue)
            .help(help);
        if let Some(short) = short {
            arg = arg.short(short);
        }
        cmd = cmd.arg(arg);
    }
    cmd
}

fn parse_qmatrix(raw: &str) -> Result<Qmatrix, String> {
    if raw == "auto" {
        return Ok(Qmatrix::Auto);
    }
    match raw.parse::<u8>() {
        Ok(level @ 0..=15) => Ok(Qmatrix::Level(level)),
        _ => Err("expected auto or 0-15".into()),
    }
}

fn encoder(matches: &ArgMatches) -> Encoder {
    match matches.get_one::<String>("encoder").unwrap().as_str() {
        "av1" => Encoder::Av1,
        "av2" => Encoder::Av2,
        "hevc" | "heic" | "heif" => Encoder::Hevc,
        "jxl" | "jpegxl" | "jpeg-xl" => Encoder::JpegXl,
        "vvc" => Encoder::Vvc,
        _ => unreachable!(),
    }
}

fn normalize_args(args: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
    let mut positional_only = false;
    args.into_iter()
        .map(|arg| {
            if arg == "--" {
                positional_only = true;
            }
            if !positional_only && arg == "-override" {
                "--override".into()
            } else {
                arg
            }
        })
        .collect()
}

pub fn parse_args() -> Args {
    let matches = command().get_matches_from(normalize_args(std::env::args_os()));
    let selected = encoder(&matches);
    if matches.get_flag("encoder-version") {
        println!("{}", version::encoder(selected));
        std::process::exit(0);
    }
    if matches.get_flag("list-overrides") {
        if selected == Encoder::Hevc && !cfg!(feature = "heic") {
            command()
                .error(
                    ErrorKind::InvalidValue,
                    "HEIC overrides require a build with --features heic",
                )
                .exit();
        }
        println!(
            "{}\n\n{}",
            match selected {
                Encoder::Av1 => crate::overrides::AV1_OPTIONS,
                Encoder::JpegXl => crate::overrides::JXL_OPTIONS,
                Encoder::Hevc => crate::overrides::HEIC_OPTIONS,
                _ => command()
                    .error(
                        ErrorKind::InvalidValue,
                        "overrides are supported for av1, jxl, and heic"
                    )
                    .exit(),
            },
            crate::overrides::COMMON_OPTIONS,
        );
        std::process::exit(0);
    }
    args_from_matches(&matches).unwrap_or_else(|e| e.exit())
}

fn args_from_matches(matches: &ArgMatches) -> Result<Args, clap::Error> {
    let encoder = encoder(matches);
    if encoder == Encoder::Hevc && !cfg!(feature = "heic") {
        return Err(command().error(
            ErrorKind::InvalidValue,
            "HEVC support is disabled; rebuild with --features heic",
        ));
    }
    if encoder == Encoder::Vvc && !cfg!(feature = "vvc") {
        return Err(command().error(
            ErrorKind::InvalidValue,
            "VVC support is disabled; rebuild with --features vvc",
        ));
    }
    let input = matches.get_one::<PathBuf>("input").unwrap().clone();
    let output = matches
        .get_one::<PathBuf>("output")
        .cloned()
        .unwrap_or_else(|| {
            input.with_extension(match encoder {
                Encoder::Av1 | Encoder::Av2 => "avif",
                Encoder::JpegXl => "jxl",
                Encoder::Hevc => "heic",
                Encoder::Vvc => "heif",
            })
        });
    if input == output {
        return Err(command().error(
            ErrorKind::InvalidValue,
            "input and output paths must differ; specify an output filename",
        ));
    }
    let lossless = matches.get_flag("lossless");
    let chroma = match matches.get_one::<String>("chroma").map(String::as_str) {
        Some("444") => Chroma::C444,
        Some("422") => Chroma::C422,
        Some("420") => Chroma::C420,
        None if lossless => Chroma::C444,
        None => Chroma::C422,
        _ => unreachable!(),
    };
    let overrides = matches
        .get_many::<OverrideGroup>("override")
        .map(|groups| {
            groups
                .flat_map(|group| group.0.iter().cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for option in &overrides {
        if option.encoder != encoder {
            return Err(command().error(ErrorKind::ArgumentConflict, format!("override '{option}' does not match the selected encoder; select it with --encoder")));
        }
    }
    if !overrides.is_empty() && crate::raster::raster_output_format(&output).is_some() {
        return Err(command().error(
            ErrorKind::ArgumentConflict,
            "encoder overrides cannot be applied to raster output; use an AVIF, JXL, or HEIC output path",
        ));
    }
    let threads = *matches.get_one::<usize>("threads").unwrap();
    let mut parsed = Args {
        input,
        output,
        encoder,
        quality: matches
            .get_one::<u8>("quality")
            .copied()
            .unwrap_or(match encoder {
                Encoder::Av1 | Encoder::Hevc => 80,
                Encoder::Av2 | Encoder::Vvc => 60,
                Encoder::JpegXl => 70,
            }),
        lossless,
        chroma: Some(chroma),
        depth: matches
            .get_one::<String>("depth")
            .map(|v| match v.as_str() {
                "8" => Depth::D8,
                "10" => Depth::D10,
                "12" => Depth::D12,
                "16" => Depth::D16,
                _ => unreachable!(),
            }),
        threads: if threads == 0 {
            basic_concurrency()
        } else {
            threads
        },
        no_alpha: matches.get_flag("no-alpha"),
        no_exif: matches.get_flag("no-exif"),
        no_icc: matches.get_flag("no-icc"),
        no_gain_map: false,
        apply_icc: matches.get_flag("apply-icc"),
        verbose: matches.get_flag("verbose"),
        speed: match matches.get_one::<String>("speed").unwrap().as_str() {
            "extraslow" => EncodingEffort::ExtraSlow,
            "slow" => EncodingEffort::Slow,
            "medium" => EncodingEffort::Medium,
            "fast" => EncodingEffort::Fast,
            _ => unreachable!(),
        },
        qmatrix: matches.get_one::<Qmatrix>("qm").copied(),
        updating_cdf: !matches.get_flag("no-cdf-update"),
        screen_content: !matches.get_flag("no-screen-content"),
        intrabc: !matches.get_flag("no-intrabc"),
        cdef: matches.get_flag("cdef"),
        wiener: matches.get_flag("wiener"),
        overrides: Vec::new(),
    };
    for option in &overrides {
        option
            .apply_pipeline(&mut parsed)
            .map_err(|e| command().error(ErrorKind::InvalidValue, e))?;
    }
    if !matches.contains_id("chroma") && !overrides.iter().any(|option| option.key() == "chroma") {
        parsed.chroma = Some(if parsed.lossless {
            Chroma::C444
        } else {
            Chroma::C422
        });
    }
    let lossless_ycbcr = overrides
        .iter()
        .rev()
        .find_map(|option| option.lossless_ycbcr())
        .unwrap_or(false);
    if parsed.lossless
        && parsed.chroma != Some(Chroma::C444)
        && encoder != Encoder::JpegXl
        && !lossless_ycbcr
    {
        return Err(command().error(
            ErrorKind::ArgumentConflict,
            "lossless RGB requires --chroma 444; HEIC lossless-ycbcr=true permits subsampling",
        ));
    }
    parsed.overrides = overrides;
    if parsed.depth == Some(Depth::D16) && encoder != Encoder::JpegXl {
        return Err(command().error(
            ErrorKind::InvalidValue,
            "16-bit output is supported only by JXL; use depth 8, 10, or 12 for this encoder",
        ));
    }
    Ok(parsed)
}

#[cfg(test)]
pub(crate) fn try_parse(args: &[&str]) -> Result<Args, clap::Error> {
    let matches =
        command().try_get_matches_from(normalize_args(args.iter().map(OsString::from)))?;
    args_from_matches(&matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_profiles_and_choose_encoder_extension() {
        for (backend, extension, quality) in
            [("av1", "avif", 80), ("av2", "avif", 60), ("jxl", "jxl", 70)]
        {
            let args = try_parse(&["cli", "-e", backend, "dir/photo.edit.png"]).unwrap();
            assert_eq!(
                args.output,
                PathBuf::from(format!("dir/photo.edit.{extension}"))
            );
            assert_eq!(args.quality, quality);
            assert!(!args.no_icc && !args.apply_icc && !args.no_exif);
        }
    }

    #[test]
    fn pipeline_overrides_select_lossless_layout_and_metadata_before_encoding() {
        for encoder in ["av1", "jxl"] {
            let group = format!(
                "{encoder}:lossless=true,depth=12,alpha=false,icc=false,exif=false,gain-map=false"
            );
            let args =
                try_parse(&["cli", "-e", encoder, "-c", "444", "-O", &group, "in.png"]).unwrap();
            assert!(args.lossless);
            assert_eq!(args.depth, Some(Depth::D12));
            assert!(args.no_alpha && args.no_icc && args.no_exif && args.no_gain_map);
        }
        let args = try_parse(&["cli", "-O", "av1:lossless=true", "in.png"]).unwrap();
        assert_eq!(args.chroma, Some(Chroma::C444));
        let args = try_parse(&[
            "cli",
            "--lossless",
            "-O",
            "av1:lossless=false,chroma=420,depth=auto",
            "in.png",
        ])
        .unwrap();
        assert!(!args.lossless);
        assert_eq!(args.chroma, Some(Chroma::C420));
        assert_eq!(args.depth, None);
        assert!(try_parse(&["cli", "-O", "av1:lossless=true,chroma=420", "in.png"]).is_err());
    }

    #[cfg(feature = "heic")]
    #[test]
    fn heic_aliases_share_overrides_and_validate_the_final_lossless_domain() {
        for encoder in ["heic", "hevc", "heif"] {
            let args = try_parse(&[
                "cli",
                "-e",
                encoder,
                "-O",
                "heic:lossless=true,lossless-ycbcr=true,chroma=420,depth=10",
                "in.png",
            ])
            .unwrap();
            assert!(args.lossless);
            assert_eq!(args.encoder, Encoder::Hevc);
            assert_eq!(args.chroma, Some(Chroma::C420));
            assert_eq!(args.depth, Some(Depth::D10));
        }
        assert!(
            try_parse(&[
                "cli",
                "-e",
                "heic",
                "-O",
                "hevc:lossless=true,chroma=420,lossless-ycbcr=true,lossless-ycbcr=false",
                "in.png"
            ])
            .is_err()
        );
    }

    #[test]
    fn sixteen_bit_depth_is_available_for_jxl() {
        let args = try_parse(&[
            "cli",
            "-e",
            "jxl",
            "-O",
            "jxl:lossless=true,depth=16",
            "in.png",
        ])
        .unwrap();
        assert_eq!(args.depth, Some(Depth::D16));
        assert!(try_parse(&["cli", "--depth", "16", "in.png"]).is_err());
    }

    #[cfg(not(feature = "heic"))]
    #[test]
    fn disabled_heic_overrides_fail_with_a_feature_hint() {
        let error =
            try_parse(&["cli", "-e", "heic", "-O", "heic:sao=false", "in.png"]).unwrap_err();
        assert!(error.to_string().contains("--features heic"));
    }

    #[test]
    fn supports_repeated_overrides_and_original_spelling() {
        let args = try_parse(&[
            "cli",
            "-e",
            "jxl",
            "-override",
            "jxl:-d:31",
            "-O",
            "jxl:patches=true",
            "in.png",
        ])
        .unwrap();
        assert_eq!(args.overrides.len(), 2);
    }

    #[test]
    fn grouped_and_repeated_overrides_preserve_order_and_last_value_wins() {
        let args = try_parse(&[
            "cli",
            "-e",
            "jxl",
            "-O",
            "jxl:distance=9,speed=medium,progressive=true",
            "--override",
            "jxl:distance=3.1,threads=1,distance=2",
            "-O",
            "jxl:distance=1.5",
            "in.png",
        ])
        .unwrap();
        assert_eq!(args.overrides.len(), 7);
        let mut cfg = jixel::EncodeConfig::default();
        for option in &args.overrides[..6] {
            option.apply_jxl(&mut cfg).unwrap();
        }
        assert_eq!(cfg.distance, 2.0);
        args.overrides[6].apply_jxl(&mut cfg).unwrap();
        assert_eq!(cfg.distance, 1.5);
        assert_eq!(cfg.speed, jixel::Speed::Medium);
        assert!(cfg.progressive);
        assert_eq!(cfg.num_threads, 1);
    }

    #[test]
    fn av1_groups_accept_spaces_and_attached_flag_values() {
        let args = try_parse(&[
            "cli",
            "--override=av1:quality=70, adaptive-quant = false, cdef=true, qm=7",
            "in.png",
        ])
        .unwrap();
        let mut cfg = maroontree::EncodeConfig::new();
        for option in &args.overrides {
            option.apply_av1(&mut cfg).unwrap();
        }
        assert_eq!(cfg.quality, 70);
        assert!(!cfg.adaptive_quant);
        assert!(cfg.cdef);
        assert_eq!(cfg.qmatrix_level, Some(7));
    }

    #[test]
    fn rejects_mistakes_before_opening_input() {
        for options in [
            vec!["--typo"],
            vec!["--threads"],
            vec!["--quality", "0"],
            vec!["--quality", "101"],
            vec!["--qm", "16"],
            vec!["--override", "jxl:distance=3"],
            vec!["--override", "jxl:distance=3,speed=fast"],
            vec!["--override", "av1:cdef=true,qm=16"],
            vec!["--lossless", "--chroma", "420"],
            vec!["--override", "av1:cdef=true", "out.png"],
        ] {
            let mut args = vec!["cli", "in.png"];
            args.extend(options);
            assert!(try_parse(&args).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn handles_help_version_and_encoder_information_without_input() {
        for flag in ["--help", "--version", "-V"] {
            let error = command().try_get_matches_from(["cli", flag]).unwrap_err();
            assert!(matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ));
        }
        for flag in ["--encoder-version", "--list-overrides"] {
            assert!(
                command()
                    .try_get_matches_from(["cli", "-e", "jxl", flag])
                    .is_ok()
            );
        }
    }

    #[test]
    fn supports_dash_prefixed_paths_after_separator() {
        let args = try_parse(&["cli", "--", "-override"]).unwrap();
        assert_eq!(args.input, PathBuf::from("-override"));
    }
}
