use crate::{Args, Chroma, Depth, Encoder, PngCicp};
use std::str::FromStr;

mod av1;
#[cfg(feature = "heic")]
mod heic;
mod jxl;

pub const COMMON_OPTIONS: &str = "\
Shared settings:
  lossless              boolean; selects the actual lossless encoding path
  depth (bit-depth)     auto|8|10|12; JXL also supports 16
  chroma (c)            444|422|420 (AV1/HEIC only; gray input stays monochrome)
  alpha                 boolean; retain/discard source alpha
  icc                   boolean; retain/discard the source ICC profile
  exif                  boolean; retain/discard source EXIF
  apply-icc             boolean; convert source pixels to sRGB, then remove ICC
  gain-map              boolean; retain/discard an imported HDR gain map
  orientation           1..8 (EXIF display orientation; pixels are not rotated)
                        normal|flip-h|rotate-180|flip-v|transpose|rotate-90|transverse|rotate-270
  color-encoding        srgb|linear-srgb|display-p3|linear-p3|bt2020-pq|bt2020-hlg|linear-bt2020
                        declares source pixel colors; no CMS conversion

Group with commas, repeat -O as needed, and use brackets for list values.
Settings apply after common CLI flags; later values win. Mode-specific controls
follow the encoder API (e.g. near-lossless requires lossless mode).";

pub const JXL_OPTIONS: &str = "\
jxl overrides (jixel):
  Syntax: -O 'jxl:KEY=VALUE,KEY=VALUE,...' (repeatable)

  distance (d)           positive finite distance, e.g. 3.1 or 31
  quality (q)            1..100; overrides distance if given later
  alpha-distance        auto|0..25; 0 keeps alpha lossless
  speed (s)             ultrafast|fastest|fast|medium|mediumslow|slow|extraslow
  threads (t)           nonnegative integer; 0 uses all available cores
  decoding-speed        fastest|fast|slow (lossless)
  progressive           boolean
  progressive-passes    auto|1..4 (lossy; clears an explicit shift schedule)
  progressive-shifts    auto|[SHIFT,...]; 1..4 shifts in 0..3, ending in 0
  progressive-dc        boolean (lossy)
  progressive-dc-levels  auto|positive integer
  patches               boolean
  splines               boolean; experimental spline detection
  dots                  boolean; experimental spot detection
  grain                 off|photon:ISO|measured:STRENGTH (lossy VarDCT)
  photon-noise-iso      nonnegative finite ISO; selects photon grain
  grain-strength       nonnegative finite multiplier; selects measured grain
  grain-lut            [8 finite values in 0..1]; selects explicit grain
  fill-invisible        boolean (lossy)
  learned-rate          boolean
  near-lossless         0..65535 (requires --lossless)
  upsampling            boolean (lossless)
  coding-transform      xyb|ycbcr|rgb (lossy)
  lossy-modular         off|auto|force
  group-order           scanline|center-first|diagonals
  arbiter               ssimulacra2|cvvdp
  ink-layer             boolean (lossy)
  vector-layers         boolean (lossy)
  primaries             srgb|bt709|bt2020|p3|display-p3|dci-p3
  transfer-function     srgb|linear|bt709|pq|smpte2084|hlg|smpte428
  white-point           d65|e|dci
  rendering-intent      perceptual|relative|saturation|absolute
  intensity-target      auto|positive finite nits (maximum 65504)
  min-nits              nonnegative finite nits (maximum 65504)
  relative-to-max-display boolean
  linear-below          nonnegative finite value (maximum 65504)";

pub const AV1_OPTIONS: &str = "\
av1 overrides (maroontree):
  Syntax: -O 'av1:KEY=VALUE,KEY=VALUE,...' (repeatable)

  quality (q)            1..100
  speed (s)             extraslow|slow|medium|fast
  threads (t)           nonnegative integer; 0 uses all available cores
  updating-cdf          boolean
  adaptive-quant        boolean (lossy)
  variance-boost        boolean (lossy; requires adaptive-quant)
  dark-aq               boolean (lossy; requires adaptive-quant)
  screen-content        boolean
  intrabc               boolean
  cdef                  boolean (lossy; extraslow always enables CDEF)
  wiener                boolean (lossy)
  quantization-matrices boolean (lossy)
  qmatrix-level (qm)    auto|0..15 (lossy; also enables matrices)
  max-cll               0..65535 nits (HDR content light level)
  max-fall              0..65535 nits (HDR frame-average light level)";

pub const HEIC_OPTIONS: &str = "\
heic/hevc overrides (hpvca):
  Syntax: -O 'heic:KEY=VALUE,KEY=VALUE,...' (repeatable; hevc/heif aliases accepted)

  quality (q)           1..100 (ignored in lossless mode)
  speed (s)             fast|slow
  threads (t)           nonnegative integer; 0 uses all available cores
  parallelism           single|wpp|tiles-wpp|grid|grid-wpp
  sao                   boolean; luma Sample Adaptive Offset filter
  screen-content        boolean; SCC palette/intra-block-copy (SCC-capable decoder required)
  implicit-rdpcm        boolean; lossless residual DPCM
  persistent-rice       boolean; Rice parameter adaptation
  lossless-ycbcr        boolean; lossless in YCbCr rather than bit-exact RGB
                        permits subsampled chroma in lossless mode
  max-cll               0..65535 nits (HDR content light level)
  max-fall              0..65535 nits (HDR frame-average light level)";

#[derive(Debug, Clone)]
pub struct EncoderOverride {
    pub encoder: Encoder,
    key: String,
    value: String,
}

enum PipelineSetting {
    Lossless(bool),
    Chroma(Chroma),
    Depth(Option<Depth>),
    Alpha(bool),
    Icc(bool),
    Exif(bool),
    ApplyIcc(bool),
    GainMap(bool),
}

#[derive(Debug, Clone)]
pub struct OverrideGroup(pub Vec<EncoderOverride>);

impl FromStr for OverrideGroup {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let (backend, settings) = raw.split_once(':').ok_or(
            "expected ENCODER:KEY=VALUE[,KEY=VALUE...], e.g. jxl:distance=3.1,speed=medium",
        )?;
        let mut entries = Vec::new();
        let mut start = 0;
        let mut in_list = false;
        for (index, ch) in settings.char_indices() {
            match ch {
                '[' if !in_list => in_list = true,
                ']' if in_list => in_list = false,
                '[' | ']' => return Err("unmatched or nested brackets in override list".into()),
                ',' if !in_list => {
                    entries.push(&settings[start..index]);
                    start = index + 1;
                }
                _ => {}
            }
        }
        if in_list {
            return Err("unclosed bracket in override list".into());
        }
        entries.push(&settings[start..]);
        let values = entries
            .into_iter()
            .map(|setting| {
                if setting.trim().is_empty() {
                    return Err("empty override setting; expected KEY=VALUE between commas".into());
                }
                format!("{backend}:{}", setting.trim()).parse()
            })
            .collect::<Result<Vec<EncoderOverride>, String>>()?;
        Ok(Self(values))
    }
}

impl std::fmt::Display for EncoderOverride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend = match self.encoder {
            Encoder::JpegXl => "jxl",
            Encoder::Hevc => "heic",
            _ => "av1",
        };
        write!(f, "{backend}:{}={}", self.key, self.value)
    }
}

impl FromStr for EncoderOverride {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let (backend, setting) = raw
            .split_once(':')
            .ok_or("expected ENCODER:KEY=VALUE, e.g. jxl:distance=3.1")?;
        let encoder = match backend.trim() {
            "jxl" | "jpegxl" | "jpeg-xl" => Encoder::JpegXl,
            "av1" => Encoder::Av1,
            "heic" | "hevc" | "heif" => Encoder::Hevc,
            _ => {
                return Err(format!(
                    "unknown override encoder '{backend}'; use av1, jxl, or heic"
                ));
            }
        };
        // Also accept the originally proposed spelling: jxl:-d:31.
        let (key, value) = setting
            .split_once('=')
            .or_else(|| setting.split_once(':'))
            .ok_or("expected ENCODER:KEY=VALUE, e.g. av1:cdef=true")?;
        let key = key.trim().trim_start_matches('-').replace('_', "-");
        let key = match key.as_str() {
            "d" if encoder == Encoder::JpegXl => "distance",
            "d" | "bit-depth" => "depth",
            "c" => "chroma",
            "q" => "quality",
            "s" => "speed",
            "t" | "num-threads" => "threads",
            "qm" => "qmatrix-level",
            "maxcll" => "max-cll",
            "maxfall" => "max-fall",
            _ => &key,
        }
        .to_owned();
        let option = Self {
            encoder,
            key,
            value: value.trim().to_owned(),
        };
        match encoder {
            Encoder::JpegXl => option.apply_jxl(&mut jixel::EncodeConfig::default())?,
            Encoder::Av1 => option.apply_av1(&mut maroontree::EncodeConfig::new())?,
            Encoder::Hevc => {
                #[cfg(feature = "heic")]
                option.apply_heic(&mut hpvca::EncodeConfig::new())?;
                #[cfg(not(feature = "heic"))]
                return Err("HEIC overrides require a build with --features heic".into());
            }
            _ => unreachable!(),
        }
        Ok(option)
    }
}

impl EncoderOverride {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn lossless_ycbcr(&self) -> Option<bool> {
        (self.encoder == Encoder::Hevc && self.key == "lossless-ycbcr")
            .then(|| self.boolean().ok())
            .flatten()
    }

    pub fn overrides_input_color(&self) -> bool {
        matches!(
            self.key.as_str(),
            "color-encoding" | "primaries" | "transfer-function" | "white-point"
        )
    }

    fn invalid(&self, expected: &str) -> String {
        format!("invalid override '{self}'; expected {expected}")
    }

    fn number<T: FromStr + PartialOrd>(&self, min: T, max: T, expected: &str) -> Result<T, String> {
        self.number_value(&self.value, min, max, expected)
    }

    fn number_value<T: FromStr + PartialOrd>(
        &self,
        raw: &str,
        min: T,
        max: T,
        expected: &str,
    ) -> Result<T, String> {
        let value = raw
            .trim()
            .parse::<T>()
            .map_err(|_| self.invalid(expected))?;
        if value >= min && value <= max {
            Ok(value)
        } else {
            Err(self.invalid(expected))
        }
    }

    fn finite(&self, min: f32, max: f32) -> Result<f32, String> {
        let expected = if max == f32::MAX {
            if min > 0.0 {
                "a positive finite number".into()
            } else {
                "a nonnegative finite number".into()
            }
        } else {
            format!("a finite number in {min}..={max}")
        };
        self.number(min, max, &expected)
    }

    fn list<T: FromStr + PartialOrd + Copy>(
        &self,
        min: T,
        max: T,
        min_len: usize,
        max_len: usize,
        expected: &str,
    ) -> Result<Vec<T>, String> {
        let raw = self
            .value
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| self.invalid(expected))?;
        let values = raw
            .split(',')
            .map(|v| self.number_value(v, min, max, expected))
            .collect::<Result<Vec<_>, _>>()?;
        if !(min_len..=max_len).contains(&values.len()) {
            return Err(self.invalid(expected));
        }
        Ok(values)
    }

    fn orientation(&self) -> Result<u8, String> {
        match self.value.as_str() {
            "normal" => Ok(1), "flip-h" => Ok(2), "rotate-180" => Ok(3),
            "flip-v" => Ok(4), "transpose" => Ok(5), "rotate-90" => Ok(6),
            "transverse" => Ok(7), "rotate-270" => Ok(8),
            _ => self.number(1, 8, "EXIF orientation 1..8 or normal|flip-h|rotate-180|flip-v|transpose|rotate-90|transverse|rotate-270"),
        }
    }

    fn source_color(&self) -> Result<PngCicp, String> {
        let (primaries, transfer) = match self.value.as_str() {
            "srgb" => (1, 13),
            "linear-srgb" => (1, 8),
            "display-p3" => (12, 13),
            "linear-p3" => (12, 8),
            "bt2020-pq" => (9, 16),
            "bt2020-hlg" => (9, 18),
            "linear-bt2020" => (9, 8),
            _ => {
                return Err(self.invalid(
                    "srgb|linear-srgb|display-p3|linear-p3|bt2020-pq|bt2020-hlg|linear-bt2020",
                ));
            }
        };
        Ok(PngCicp {
            color_primaries: primaries,
            transfer_function: transfer,
            matrix_coefficients: 0,
            full_range: true,
        })
    }

    fn pipeline_setting(&self) -> Result<Option<PipelineSetting>, String> {
        Ok(Some(match self.key.as_str() {
            "lossless" => PipelineSetting::Lossless(self.boolean()?),
            "depth" => PipelineSetting::Depth(match self.value.as_str() {
                "auto" => None,
                "8" => Some(Depth::D8),
                "10" => Some(Depth::D10),
                "12" => Some(Depth::D12),
                "16" if self.encoder == Encoder::JpegXl => Some(Depth::D16),
                _ => return Err(self.invalid("auto|8|10|12 (JXL also supports 16)")),
            }),
            "chroma" if self.encoder != Encoder::JpegXl => {
                PipelineSetting::Chroma(match self.value.as_str() {
                    "444" => Chroma::C444,
                    "422" => Chroma::C422,
                    "420" => Chroma::C420,
                    _ => return Err(self.invalid("444|422|420")),
                })
            }
            "alpha" => PipelineSetting::Alpha(self.boolean()?),
            "icc" => PipelineSetting::Icc(self.boolean()?),
            "exif" => PipelineSetting::Exif(self.boolean()?),
            "apply-icc" => PipelineSetting::ApplyIcc(self.boolean()?),
            "gain-map" => PipelineSetting::GainMap(self.boolean()?),
            _ => return Ok(None),
        }))
    }

    pub fn apply_pipeline(&self, args: &mut Args) -> Result<(), String> {
        match self.pipeline_setting()? {
            Some(PipelineSetting::Lossless(v)) => args.lossless = v,
            Some(PipelineSetting::Depth(v)) => args.depth = v,
            Some(PipelineSetting::Chroma(v)) => args.chroma = Some(v),
            Some(PipelineSetting::Alpha(v)) => args.no_alpha = !v,
            Some(PipelineSetting::Icc(v)) => args.no_icc = !v,
            Some(PipelineSetting::Exif(v)) => args.no_exif = !v,
            Some(PipelineSetting::ApplyIcc(v)) => args.apply_icc = v,
            Some(PipelineSetting::GainMap(v)) => args.no_gain_map = !v,
            None => {}
        }
        Ok(())
    }

    fn boolean(&self) -> Result<bool, String> {
        match self.value.as_str() {
            "true" | "1" | "on" => Ok(true),
            "false" | "0" | "off" => Ok(false),
            _ => Err(self.invalid("true or false (also 1/0 or on/off)")),
        }
    }

    fn threads(&self) -> Result<usize, String> {
        let n = self.number(0, usize::MAX, "a nonnegative integer")?;
        Ok(if n == 0 {
            crate::basic_concurrency()
        } else {
            n
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracketed_jxl_lists_stay_inside_their_grouped_setting() {
        let group = "jxl:progressive-shifts=[2, 1, 0],grain-lut=[0,0.01,0.02,0.03,0.04,0.05,0.06,0.07],splines=true,dots=true,orientation=rotate-90"
            .parse::<OverrideGroup>().unwrap();
        assert_eq!(group.0.len(), 5);
        let mut cfg = jixel::EncodeConfig::default();
        for option in &group.0 {
            option.apply_jxl(&mut cfg).unwrap();
        }
        assert_eq!(cfg.progressive_shifts, Some(vec![2, 1, 0]));
        assert_eq!(
            cfg.grain,
            jixel::Grain::Lut([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        );
        assert!(cfg.splines && cfg.dots);
        assert_eq!(cfg.orientation, jixel::Orientation::Rotate90);
        "jxl:progressive-passes=2"
            .parse::<EncoderOverride>()
            .unwrap()
            .apply_jxl(&mut cfg)
            .unwrap();
        assert_eq!(cfg.progressive_shifts, None);
        assert_eq!(cfg.progressive_passes, Some(2));
    }

    #[test]
    fn jxl_grain_modes_and_color_controls_reach_the_configuration() {
        let mut cfg = jixel::EncodeConfig::default();
        for (setting, grain) in [
            ("grain=photon:3200", jixel::Grain::PhotonIso(3200.0)),
            ("grain=measured:1.5", jixel::Grain::Measured(1.5)),
            ("grain=off", jixel::Grain::Off),
        ] {
            format!("jxl:{setting}")
                .parse::<EncoderOverride>()
                .unwrap()
                .apply_jxl(&mut cfg)
                .unwrap();
            assert_eq!(cfg.grain, grain);
        }
        for option in "jxl:color-encoding=bt2020-pq,primaries=display-p3,transfer-function=hlg,white-point=d65,rendering-intent=absolute".parse::<OverrideGroup>().unwrap().0 {
            option.apply_jxl(&mut cfg).unwrap();
        }
        assert_eq!(cfg.color_encoding.primaries, jixel::Primaries::Smpte431);
        assert_eq!(cfg.color_encoding.transfer, jixel::TransferFunction::Hlg);
        assert_eq!(cfg.color_encoding.white_point, jixel::WhitePoint::D65);
        assert_eq!(
            cfg.color_encoding.rendering_intent,
            jixel::RenderingIntent::Absolute
        );
    }

    #[test]
    fn av1_color_overrides_preserve_the_coded_matrix_and_range() {
        let mut cfg = maroontree::EncodeConfig::new().with_cicp(maroontree::Cicp {
            matrix: maroontree::MatrixCoefficients::YCgCo,
            full_range: true,
            ..maroontree::Cicp::srgb_ycbcr()
        });
        for option in "av1:color-encoding=bt2020-pq,orientation=6,max-cll=1000,max-fall=400"
            .parse::<OverrideGroup>()
            .unwrap()
            .0
        {
            option.apply_av1(&mut cfg).unwrap();
        }
        let color = cfg.color_encoding.unwrap();
        assert_eq!(color.primaries, maroontree::Primaries::Bt2020);
        assert_eq!(color.transfer, maroontree::TransferFunction::Smpte2084);
        assert_eq!(color.matrix, maroontree::MatrixCoefficients::YCgCo);
        assert!(color.full_range);
        assert_eq!(cfg.metadata.orientation, maroontree::Orientation::Rotate90);
        assert_eq!(
            cfg.metadata.content_light_level,
            Some(maroontree::ContentLightLevel::new(1000, 400))
        );
    }

    #[cfg(feature = "heic")]
    #[test]
    fn heic_controls_reach_the_encoder_configuration() {
        let mut cfg = hpvca::EncodeConfig::new();
        for option in "heif:quality=81,speed=slow,threads=2,parallelism=grid-wpp,sao=false,screen-content=true,implicit-rdpcm=true,persistent-rice=false,lossless-ycbcr=true,orientation=8,max-cll=1000,max-fall=400,color-encoding=display-p3".parse::<OverrideGroup>().unwrap().0 {
            option.apply_heic(&mut cfg).unwrap();
        }
        assert_eq!(cfg.quality, 81);
        assert_eq!(cfg.speed, hpvca::Speed::Slow);
        assert_eq!(cfg.threads, 2);
        assert_eq!(cfg.parallelism, hpvca::ParallelismStrategy::GridWpp);
        assert!(!cfg.sao && !cfg.persistent_rice);
        assert!(cfg.screen_content && cfg.implicit_rdpcm && cfg.lossless_ycbcr);
        assert_eq!(cfg.metadata.orientation, hpvca::Orientation::Rotate270);
        assert_eq!(
            cfg.metadata.content_light_level,
            Some(hpvca::ContentLightLevel::new(1000, 400))
        );
        assert_eq!(
            cfg.color.cicp.unwrap().primaries,
            hpvca::Primaries::Smpte432
        );
    }

    #[test]
    fn invalid_lists_metadata_and_mode_values_are_rejected() {
        for raw in [
            "jxl:progressive-shifts=[2,1",
            "jxl:progressive-shifts=2,1,0",
            "jxl:progressive-shifts=[[2,1,0]]",
            "jxl:progressive-shifts=[2,1]",
            "jxl:progressive-shifts=[4,0]",
            "jxl:progressive-shifts=[3,2,1,1,0]",
            "jxl:grain-lut=[0,1]",
            "jxl:grain-lut=[0,0,0,0,0,0,0,NaN]",
            "jxl:grain=photon:inf",
            "jxl:grain=measured:-1",
            "av1:depth=16",
            "jxl:chroma=420",
            "jxl:orientation=9",
            "jxl:color-encoding=unknown",
            "av1:alpha=maybe",
            "av1:max-cll=65536",
            "av1:depth=9",
        ] {
            assert!(raw.parse::<OverrideGroup>().is_err(), "accepted {raw}");
        }
        #[cfg(feature = "heic")]
        for raw in [
            "heic:parallelism=unknown",
            "heic:sao=maybe",
            "heic:quality=101",
            "heic:speed=medium",
            "heic:max-fall=-1",
        ] {
            assert!(raw.parse::<OverrideGroup>().is_err(), "accepted {raw}");
        }
    }

    #[test]
    fn groups_support_whitespace_and_legacy_parameter_names() {
        let group = " jxl : -d : 31, -s : fast, alpha_distance = 0 "
            .parse::<OverrideGroup>()
            .unwrap();
        let mut cfg = jixel::EncodeConfig::default();
        for option in &group.0 {
            option.apply_jxl(&mut cfg).unwrap();
        }
        assert_eq!(cfg.distance, 31.0);
        assert_eq!(cfg.speed, jixel::Speed::Fast);
        assert_eq!(cfg.alpha_distance, Some(0.0));
    }

    #[test]
    fn rejects_an_entire_group_if_any_setting_is_invalid() {
        for raw in [
            "jxl",
            "jxl:",
            "jxl:,distance=3",
            "jxl:distance=3,",
            "jxl:distance=3,,speed=fast",
            "jxl:distance=3,speed=typo",
            "jxl:distance=3,progressive=maybe",
            "av1:cdef=true,qm=16",
            "av1:cdef=true,quality=",
            "av1:cdef=true,unknown=1",
        ] {
            assert!(raw.parse::<OverrideGroup>().is_err(), "accepted {raw}");
        }
    }

    #[test]
    fn accepts_both_syntaxes_and_keeps_last_override() {
        let mut cfg = jixel::EncodeConfig::default();
        for raw in ["jxl:distance=3.1", "jxl:-d:31", "jxl:alpha_distance=0"] {
            raw.parse::<EncoderOverride>()
                .unwrap()
                .apply_jxl(&mut cfg)
                .unwrap();
        }
        assert_eq!(cfg.distance, 31.0);
        assert_eq!(cfg.alpha_distance, Some(0.0));
    }

    #[test]
    fn av1_overrides_reach_encoder_config() {
        let mut cfg = maroontree::EncodeConfig::new().with_cdef(true);
        for raw in [
            "av1:cdef=false",
            "av1:adaptive-quant=0",
            "av1:qm=7",
            "av1:quality=92",
            "av1:threads=1",
            "av1:speed=fast",
        ] {
            raw.parse::<EncoderOverride>()
                .unwrap()
                .apply_av1(&mut cfg)
                .unwrap();
        }
        assert!(!cfg.cdef);
        assert!(!cfg.adaptive_quant);
        assert_eq!(cfg.qmatrix_level, Some(7));
        assert_eq!(cfg.quality, 92);
        assert_eq!(cfg.threads, 1);
        assert_eq!(cfg.speed, maroontree::Speed::Fast);
    }

    #[test]
    fn rejects_unknown_keys_bad_syntax_and_invalid_values() {
        for raw in [
            "jxl",
            "jxl:distance",
            "vvc:quality=80",
            "jxl:typo=1",
            "jxl:distance=NaN",
            "jxl:distance=inf",
            "jxl:distance=0",
            "jxl:distance=-1",
            "jxl:alpha-distance=26",
            "jxl:progressive-passes=5",
            "av1:quality=0",
            "av1:qm=16",
            "av1:cdef=maybe",
            "av1:threads=-1",
            "av1:speed=fastest",
            "av1:quality=",
        ] {
            assert!(raw.parse::<EncoderOverride>().is_err(), "accepted {raw}");
        }
    }
}
