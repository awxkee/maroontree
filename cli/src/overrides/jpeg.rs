use super::{EncoderOverride, PipelineSetting};
use crate::jpeg::JpegOptions;
use jhm17::{
    ChromaSubsamplingMethod, Polish, PolishJudge, ProgressiveMode, QuantizationTableType,
    SamplePrecision, SamplingFactor,
};

pub const OPTIONS: &str = "\
jpeg/jpg/jhm17 overrides (jhm17; polishing compiled in and enabled by default):
  Syntax: -O 'jpeg:KEY=VALUE,KEY=VALUE,...' (repeatable)

  quality (q)            1..100 [default: 80]
  chroma (c)             420|444|422 [default: 420; 422 requires polish=off]
  sampling-factor        444|440|441|422|420|421|411|410
  subsampling-method     average|nearest
  sample-precision       all|luma|integer
  speed (s)              fast|medium|slow|extraslow; selects greedy|trellis|trellis-greedy polishing
  progressive            false|true|off|optimized|standard
  progressive-scans      2..64; selects spectral progressive encoding
  restart-interval       0..65535 MCUs; 0 disables restart markers
  optimize-huffman       boolean
  entropy                huffman|arithmetic
  arithmetic             boolean (arithmetic JPEG needs a compatible decoder)
  quantization-table     default|flat|ms-ssim|psnr-hvs|image-magick|klein-silverstein-carney|dental-xrays|visual-detection|improved-detection
  luma-table/chroma-table table preset or [64 positive u16 values]
  luma-dc/luma-ac         positive finite step multipliers
  chroma-dc/chroma-ac     step multipliers for both chroma channels
  cb-dc/cb-ac/cr-dc/cr-ac  step multipliers for individual chroma channels
  luma-bands/chroma-bands/cb-bands/cr-bands [5 positive finite band multipliers]
  auto-chroma-scale      boolean; refine chroma steps for saturated reds
  deadzone-luma/deadzone-chroma/deadzone-chroma-full [4 nonnegative finite thresholds]

RDO (enabled by default; setting an rdo-* parameter enables RDO):
  rdo                    boolean
  rdo-lambda             nonnegative finite rate price; 0 disables RDO
  rdo-chroma-weight/rdo-chroma-weight-full nonnegative finite distortion weights
  rdo-distortion-power   0..1
  rdo-tile-weights       0..1
  rdo-tile-chroma/rdo-tile-chroma-full boolean
  rdo-aq-strength        nonnegative finite strength
  rdo-aq-min/rdo-aq-max   positive finite scales; min must not exceed max
  rdo-aq-chroma          boolean
  rdo-aq-brightness      finite signed exponent
  rdo-aq-brightness-max  positive finite upper limit
  rdo-clip-iterations    0..255; 0 disables white-pixel clip continuation

Polishing (RGB 420/444; grayscale uses the configured quantizer):
  polish                 off|greedy|trellis|trellis-greedy [default: trellis-greedy]
  polish-judge           both|ssimulacra2|cvvdp
  polish-proposals/polish-rounds nonnegative integer
  polish-max-bits        signed i32 candidate bit-cost limit
  polish-restore-threshold nonnegative finite threshold
  polish-scan-limit      0..63 (last zigzag coefficient)
  polish-cvvdp-tries     nonnegative integer
  polish-quality-step   1..99 (exchange-rate quality distance)
  polish-trellis-k/polish-trellis-proposals nonnegative integer
  Explicit polish-* settings select custom polishing and enable it.

Metadata and format:
  density                [X,Y], positive u16 density values
  density-unit           aspect|dpi|dpcm
  icc/exif/apply-icc      boolean; same behavior as the common CLI flags
  depth                  auto|8; JPEG output is 8-bit and has no alpha
  alpha/gain-map/lossless only false is supported
  JPEG encoding is single-threaded. Groups and repeated flags apply in order;
  later values win. Polish requires RGB 420/444 and adds encoding time.";

impl EncoderOverride {
    pub fn apply_jpeg(&self, cfg: &mut JpegOptions) -> Result<(), String> {
        if let Some(setting) = self.pipeline_setting()? {
            return match setting {
                PipelineSetting::Lossless(true) => Err(self.invalid(
                    "lossless=false; JPEG does not support pixel-perfect lossless encoding",
                )),
                PipelineSetting::Depth(Some(depth)) if depth != crate::Depth::D8 => {
                    Err(self.invalid("auto or 8; jhm17 JPEG output is 8-bit"))
                }
                PipelineSetting::Alpha(true) => {
                    Err(self.invalid("alpha=false; JPEG has no alpha channel"))
                }
                PipelineSetting::GainMap(true) => {
                    Err(self
                        .invalid("gain-map=false; this JPEG backend does not encode HDR gain maps"))
                }
                PipelineSetting::Chroma(chroma) => {
                    cfg.sampling = match chroma {
                        crate::Chroma::C444 => SamplingFactor::R_4_4_4,
                        crate::Chroma::C422 => SamplingFactor::R_4_2_2,
                        crate::Chroma::C420 => SamplingFactor::R_4_2_0,
                    };
                    Ok(())
                }
                _ => Ok(()),
            };
        }
        match self.key.as_str() {
            "quality" => cfg.quality = self.number(1, 100, "1..100")?,
            "sampling" | "sampling-factor" => {
                cfg.sampling = match self.value.as_str() {
                    "444" => SamplingFactor::R_4_4_4,
                    "440" => SamplingFactor::R_4_4_0,
                    "441" => SamplingFactor::R_4_4_1,
                    "422" => SamplingFactor::R_4_2_2,
                    "420" => SamplingFactor::R_4_2_0,
                    "421" => SamplingFactor::R_4_2_1,
                    "411" => SamplingFactor::R_4_1_1,
                    "410" => SamplingFactor::R_4_1_0,
                    _ => return Err(self.invalid("444|440|441|422|420|421|411|410")),
                }
            }
            "subsampling-method" => {
                cfg.subsampling = match self.value.as_str() {
                    "average" => ChromaSubsamplingMethod::Average,
                    "nearest" => ChromaSubsamplingMethod::Nearest,
                    _ => return Err(self.invalid("average|nearest")),
                }
            }
            "sample-precision" => {
                cfg.precision = match self.value.as_str() {
                    "all" => SamplePrecision::All,
                    "luma" => SamplePrecision::Luma,
                    "integer" => SamplePrecision::Integer,
                    _ => return Err(self.invalid("all|luma|integer")),
                }
            }
            "speed" => {
                cfg.polish = JpegOptions::polish_for_speed(match self.value.as_str() {
                    "fast" => crate::EncodingEffort::Fast,
                    "medium" => crate::EncodingEffort::Medium,
                    "slow" => crate::EncodingEffort::Slow,
                    "extraslow" => crate::EncodingEffort::ExtraSlow,
                    _ => return Err(self.invalid("fast|medium|slow|extraslow")),
                })
            }
            "progressive" => {
                cfg.progressive = match self.value.as_str() {
                    "true" | "on" | "1" | "optimized" => Some(ProgressiveMode::Optimized),
                    "false" | "off" | "0" => None,
                    "standard" => Some(ProgressiveMode::Standard),
                    _ => return Err(self.invalid("false|true|off|optimized|standard")),
                }
            }
            "progressive-scans" => {
                cfg.progressive = Some(ProgressiveMode::Spectral(self.number(2, 64, "2..64")?))
            }
            "restart-interval" => cfg.restart_interval = self.number(0, u16::MAX, "0..65535")?,
            "optimize-huffman" | "optimized-huffman-tables" => {
                cfg.optimize_huffman = self.boolean()?
            }
            "arithmetic" | "arithmetic-coding" => cfg.arithmetic = self.boolean()?,
            "entropy" => {
                cfg.arithmetic = match self.value.as_str() {
                    "huffman" => false,
                    "arithmetic" => true,
                    _ => return Err(self.invalid("huffman|arithmetic")),
                }
            }
            "quantization-table" => {
                let table = self.jpeg_table()?;
                cfg.luma_table = table.clone();
                cfg.chroma_table = table;
            }
            "luma-table" => cfg.luma_table = self.jpeg_table()?,
            "chroma-table" => cfg.chroma_table = self.jpeg_table()?,
            "auto-chroma-scale" => cfg.auto_chroma_scale = self.boolean()?,
            "deadzone-luma" => {
                cfg.deadzone.luma = self
                    .list(0.0, f32::MAX, 4, 4, "[4 nonnegative finite thresholds]")?
                    .try_into()
                    .unwrap()
            }
            "deadzone-chroma" => {
                cfg.deadzone.chroma = self
                    .list(0.0, f32::MAX, 4, 4, "[4 nonnegative finite thresholds]")?
                    .try_into()
                    .unwrap()
            }
            "deadzone-chroma-full" => {
                cfg.deadzone.chroma_full = self
                    .list(0.0, f32::MAX, 4, 4, "[4 nonnegative finite thresholds]")?
                    .try_into()
                    .unwrap()
            }
            "luma-dc" | "luma-ac" | "chroma-dc" | "chroma-ac" | "cb-dc" | "cb-ac" | "cr-dc"
            | "cr-ac" | "luma-bands" | "chroma-bands" | "cb-bands" | "cr-bands" => {
                let (channel, field) = self.key.split_once('-').unwrap();
                match channel {
                    "luma" => self.jpeg_step(&mut cfg.luma_steps, field)?,
                    "cb" => self.jpeg_step(&mut cfg.cb_steps, field)?,
                    "cr" => self.jpeg_step(&mut cfg.cr_steps, field)?,
                    "chroma" => {
                        self.jpeg_step(&mut cfg.cb_steps, field)?;
                        self.jpeg_step(&mut cfg.cr_steps, field)?;
                    }
                    _ => unreachable!(),
                }
            }
            "rdo" => cfg.rdo_enabled = self.boolean()?,
            key if key.starts_with("rdo-") => {
                match key {
                    "rdo-lambda" => cfg.rdo.lambda = self.finite(0.0, f32::MAX)?,
                    "rdo-chroma-weight" => cfg.rdo.chroma_weight = self.finite(0.0, f32::MAX)?,
                    "rdo-chroma-weight-full" => {
                        cfg.rdo.chroma_weight_full = self.finite(0.0, f32::MAX)?
                    }
                    "rdo-distortion-power" => cfg.rdo.distortion_power = self.finite(0.0, 1.0)?,
                    "rdo-tile-weights" => cfg.rdo.tile_weights = self.finite(0.0, 1.0)?,
                    "rdo-tile-chroma" => cfg.rdo.tile_chroma = self.boolean()?,
                    "rdo-tile-chroma-full" => cfg.rdo.tile_chroma_full = self.boolean()?,
                    "rdo-aq-strength" => cfg.rdo.aq_strength = self.finite(0.0, f32::MAX)?,
                    "rdo-aq-min" => cfg.rdo.aq_min = self.finite(f32::MIN_POSITIVE, f32::MAX)?,
                    "rdo-aq-max" => cfg.rdo.aq_max = self.finite(f32::MIN_POSITIVE, f32::MAX)?,
                    "rdo-aq-chroma" => cfg.rdo.aq_chroma = self.boolean()?,
                    "rdo-aq-brightness" => {
                        cfg.rdo.aq_brightness =
                            self.number(f32::MIN, f32::MAX, "a finite signed exponent")?
                    }
                    "rdo-aq-brightness-max" => {
                        cfg.rdo.aq_brightness_max = self.finite(f32::MIN_POSITIVE, f32::MAX)?
                    }
                    "rdo-clip-iterations" => {
                        cfg.rdo.clip.iterations = self.number(0, u8::MAX, "0..255")?
                    }
                    _ => {
                        return Err(
                            self.invalid("a supported RDO setting; use -e jpeg --list-overrides")
                        );
                    }
                }
                cfg.rdo_enabled = cfg.rdo.lambda > 0.0;
            }
            "polish" => {
                cfg.polish = match self.value.as_str() {
                    "off" | "false" | "0" => Polish::Off,
                    "greedy" => Polish::Greedy,
                    "trellis" => Polish::Trellis,
                    "trellis-greedy" | "true" | "on" | "1" => Polish::TrellisGreedy,
                    _ => return Err(self.invalid("off|greedy|trellis|trellis-greedy")),
                }
            }
            key if key.starts_with("polish-") => {
                let mut polish = cfg.polish_options();
                match key {
                    "polish-judge" => {
                        polish.judge = match self.value.as_str() {
                            "both" => PolishJudge::Both,
                            "ssimulacra2" | "ss2" => PolishJudge::Ssimulacra2,
                            "cvvdp" => PolishJudge::Cvvdp,
                            _ => return Err(self.invalid("both|ssimulacra2|cvvdp")),
                        }
                    }
                    "polish-proposals" => {
                        polish.proposals = self.number(0, usize::MAX, "a nonnegative integer")?
                    }
                    "polish-rounds" => {
                        polish.rounds = self.number(0, usize::MAX, "a nonnegative integer")?
                    }
                    "polish-max-bits" => {
                        polish.max_bits =
                            self.number(i32::MIN, i32::MAX, "a signed i32 bit-cost limit")?
                    }
                    "polish-restore-threshold" => {
                        polish.restore_threshold = self.finite(0.0, f32::MAX)?
                    }
                    "polish-scan-limit" => polish.scan_limit = self.number(0, 63, "0..63")?,
                    "polish-cvvdp-tries" => {
                        polish.cvvdp_tries = self.number(0, usize::MAX, "a nonnegative integer")?
                    }
                    "polish-quality-step" => polish.quality_step = self.number(1, 99, "1..99")?,
                    "polish-trellis-k" => {
                        polish.trellis_k = self.number(0, usize::MAX, "a nonnegative integer")?
                    }
                    "polish-trellis-proposals" => {
                        polish.trellis_proposals =
                            self.number(0, usize::MAX, "a nonnegative integer")?
                    }
                    _ => {
                        return Err(self.invalid(
                            "a supported polishing setting; use -e jpeg --list-overrides",
                        ));
                    }
                }
                cfg.polish = Polish::Custom(polish);
            }
            "density" => {
                let values =
                    self.list(1u16, u16::MAX, 2, 2, "[X,Y] positive u16 density values")?;
                cfg.density.density = (values[0], values[1]);
            }
            "density-unit" => {
                cfg.density.unit = match self.value.as_str() {
                    "aspect" => jhm17::PixelDensityUnit::PixelAspectRatio,
                    "dpi" => jhm17::PixelDensityUnit::Inches,
                    "dpcm" => jhm17::PixelDensityUnit::Centimeters,
                    _ => return Err(self.invalid("aspect|dpi|dpcm")),
                }
            }
            _ => {
                return Err(format!(
                    "unknown JPEG override '{}'; use -e jpeg --list-overrides",
                    self.key
                ));
            }
        }
        Ok(())
    }

    fn jpeg_table(&self) -> Result<QuantizationTableType, String> {
        Ok(match self.value.as_str() {
            "default" => QuantizationTableType::Default,
            "flat" => QuantizationTableType::Flat,
            "ms-ssim" => QuantizationTableType::CustomMsSsim,
            "psnr-hvs" => QuantizationTableType::CustomPsnrHvs,
            "image-magick" => QuantizationTableType::ImageMagick,
            "klein-silverstein-carney" => QuantizationTableType::KleinSilversteinCarney,
            "dental-xrays" => QuantizationTableType::DentalXRays,
            "visual-detection" => QuantizationTableType::VisualDetectionModel,
            "improved-detection" => QuantizationTableType::ImprovedDetectionModel,
            value if value.starts_with('[') => QuantizationTableType::Custom(Box::new(
                self.list(
                    1u16,
                    u16::MAX,
                    64,
                    64,
                    "a table preset or [64 positive u16 values]",
                )?
                .try_into()
                .unwrap(),
            )),
            _ => {
                return Err(self.invalid("a quantization-table preset or [64 positive u16 values]"));
            }
        })
    }

    fn jpeg_step(&self, step: &mut jhm17::StepTuning, field: &str) -> Result<(), String> {
        match field {
            "dc" => step.dc = self.finite(f32::MIN_POSITIVE, f32::MAX)?,
            "ac" => step.ac = self.finite(f32::MIN_POSITIVE, f32::MAX)?,
            "bands" => {
                step.bands = self
                    .list(
                        f32::MIN_POSITIVE,
                        f32::MAX,
                        5,
                        5,
                        "[5 positive finite multipliers]",
                    )?
                    .try_into()
                    .unwrap()
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}
