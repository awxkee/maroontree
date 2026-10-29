use super::EncoderOverride;

impl EncoderOverride {
    pub fn apply_jxl(&self, cfg: &mut jixel::EncodeConfig) -> Result<(), String> {
        if self.pipeline_setting()?.is_some() {
            return Ok(());
        }
        match self.key.as_str() {
            "orientation" => {
                cfg.orientation = jixel::Orientation::from_exif(self.orientation()?).unwrap()
            }
            "color-encoding" => {
                cfg.color_encoding = crate::jxl::color_encoding_from_cicp(self.source_color()?)
                    .unwrap()
                    .0
            }
            "primaries" => {
                cfg.color_encoding.primaries = match self.value.as_str() {
                    "srgb" | "bt709" => jixel::Primaries::Bt709,
                    "bt2020" => jixel::Primaries::Bt2020,
                    "p3" | "display-p3" => {
                        cfg.color_encoding.white_point = jixel::WhitePoint::D65;
                        jixel::Primaries::Smpte431
                    }
                    "dci-p3" => {
                        cfg.color_encoding.white_point = jixel::WhitePoint::Dci;
                        jixel::Primaries::Smpte431
                    }
                    _ => return Err(self.invalid("srgb|bt709|bt2020|p3|display-p3|dci-p3")),
                }
            }
            "transfer-function" => {
                cfg.color_encoding.transfer = match self.value.as_str() {
                    "srgb" => jixel::TransferFunction::Srgb,
                    "linear" => jixel::TransferFunction::Linear,
                    "bt709" => jixel::TransferFunction::Bt709,
                    "pq" | "smpte2084" => jixel::TransferFunction::Smpte2084,
                    "hlg" => jixel::TransferFunction::Hlg,
                    "smpte428" => jixel::TransferFunction::Smpte428,
                    _ => return Err(self.invalid("srgb|linear|bt709|pq|smpte2084|hlg|smpte428")),
                }
            }
            "white-point" => {
                cfg.color_encoding.white_point = match self.value.as_str() {
                    "d65" => jixel::WhitePoint::D65,
                    "e" => jixel::WhitePoint::E,
                    "dci" => jixel::WhitePoint::Dci,
                    _ => return Err(self.invalid("d65|e|dci")),
                }
            }
            "rendering-intent" => {
                cfg.color_encoding.rendering_intent = match self.value.as_str() {
                    "perceptual" => jixel::RenderingIntent::Perceptual,
                    "relative" => jixel::RenderingIntent::Relative,
                    "saturation" => jixel::RenderingIntent::Saturation,
                    "absolute" => jixel::RenderingIntent::Absolute,
                    _ => return Err(self.invalid("perceptual|relative|saturation|absolute")),
                }
            }
            "distance" => cfg.distance = self.finite(f32::MIN_POSITIVE, f32::MAX)?,
            "quality" => {
                cfg.distance = jixel::EncodeConfig::default()
                    .with_quality(self.finite(1.0, 100.0)?)
                    .distance
            }
            "alpha-distance" => {
                cfg.alpha_distance = if self.value == "auto" {
                    None
                } else {
                    Some(self.finite(0.0, 25.0)?)
                }
            }
            "threads" => cfg.num_threads = self.threads()?,
            "speed" => {
                cfg.speed = match self.value.as_str() {
                    "ultrafast" => jixel::Speed::UltraFast,
                    "fastest" => jixel::Speed::Fastest,
                    "fast" => jixel::Speed::Fast,
                    "medium" => jixel::Speed::Medium,
                    "mediumslow" => jixel::Speed::MediumSlow,
                    "slow" => jixel::Speed::Slow,
                    "extraslow" => jixel::Speed::ExtraSlow,
                    _ => {
                        return Err(
                            self.invalid("ultrafast|fastest|fast|medium|mediumslow|slow|extraslow")
                        );
                    }
                }
            }
            "decoding-speed" => {
                cfg.decoding_speed = match self.value.as_str() {
                    "fastest" => jixel::DecodingSpeed::Fastest,
                    "fast" => jixel::DecodingSpeed::Fast,
                    "slow" => jixel::DecodingSpeed::Slow,
                    _ => return Err(self.invalid("fastest|fast|slow")),
                }
            }
            "progressive" => cfg.progressive = self.boolean()?,
            "progressive-passes" => {
                cfg.progressive_passes = if self.value == "auto" {
                    None
                } else {
                    Some(self.number(1, 4, "auto or 1..4")?)
                };
                cfg.progressive_shifts = None;
            }
            "progressive-shifts" => {
                cfg.progressive_shifts = if self.value == "auto" {
                    None
                } else {
                    let shifts = self.list(
                        0u32,
                        3,
                        1,
                        4,
                        "a bracketed list of 1..4 shifts in 0..3 ending in 0, e.g. [2,1,0]",
                    )?;
                    if shifts.last() != Some(&0) {
                        return Err(self.invalid("a progressive shift schedule ending in 0"));
                    }
                    Some(shifts)
                };
            }
            "progressive-dc" => cfg.progressive_dc = self.boolean()?,
            "progressive-dc-levels" => {
                cfg.progressive_dc_levels = if self.value == "auto" {
                    None
                } else {
                    Some(self.number(1, u32::MAX, "auto or a positive integer")?)
                }
            }
            "patches" => cfg.patches = self.boolean()?,
            "splines" => cfg.splines = self.boolean()?,
            "dots" => cfg.dots = self.boolean()?,
            "grain" => {
                cfg.grain = if self.value == "off" {
                    jixel::Grain::Off
                } else {
                    let (mode, raw) = self
                        .value
                        .split_once(':')
                        .ok_or_else(|| self.invalid("off|photon:ISO|measured:STRENGTH"))?;
                    let value = self.number_value(
                        raw,
                        0.0f32,
                        f32::MAX,
                        "a nonnegative finite grain strength or ISO",
                    )?;
                    match mode {
                        "photon" => jixel::Grain::PhotonIso(value),
                        "measured" => jixel::Grain::Measured(value),
                        _ => return Err(self.invalid("off|photon:ISO|measured:STRENGTH")),
                    }
                }
            }
            "photon-noise-iso" => cfg.grain = jixel::Grain::PhotonIso(self.finite(0.0, f32::MAX)?),
            "grain-strength" => cfg.grain = jixel::Grain::Measured(self.finite(0.0, f32::MAX)?),
            "grain-lut" => {
                cfg.grain = jixel::Grain::Lut(
                    self.list(
                        0.0f32,
                        1.0,
                        8,
                        8,
                        "a bracketed list of exactly 8 finite strengths in 0..1",
                    )?
                    .try_into()
                    .unwrap(),
                )
            }
            "fill-invisible" => cfg.fill_invisible = self.boolean()?,
            "learned-rate" => cfg.learned_rate = self.boolean()?,
            "near-lossless" => cfg.near_lossless = self.number(0, u16::MAX, "0..65535")?,
            "upsampling" => cfg.upsampling = self.boolean()?,
            "coding-transform" => {
                cfg.coding_transform = match self.value.as_str() {
                    "xyb" => jixel::CodingTransform::Xyb,
                    "ycbcr" => jixel::CodingTransform::YCbCr,
                    "rgb" => jixel::CodingTransform::Rgb,
                    _ => return Err(self.invalid("xyb|ycbcr|rgb")),
                }
            }
            "lossy-modular" => {
                cfg.lossy_modular = match self.value.as_str() {
                    "off" => jixel::LossyModular::Off,
                    "auto" => jixel::LossyModular::Auto,
                    "force" => jixel::LossyModular::Force,
                    _ => return Err(self.invalid("off|auto|force")),
                }
            }
            "group-order" => {
                cfg.group_order = match self.value.as_str() {
                    "scanline" => jixel::GroupOrder::Scanline,
                    "center-first" => jixel::GroupOrder::CenterFirst,
                    "diagonals" => jixel::GroupOrder::Diagonals,
                    _ => return Err(self.invalid("scanline|center-first|diagonals")),
                }
            }
            "arbiter" => {
                cfg.arbiter = match self.value.as_str() {
                    "ssimulacra2" => jixel::Arbiter::Ssimulacra2,
                    "cvvdp" => jixel::Arbiter::Cvvdp,
                    _ => return Err(self.invalid("ssimulacra2|cvvdp")),
                }
            }
            "ink-layer" => cfg.ink_layer = self.boolean()?,
            "vector-layers" => cfg.vector_layers = self.boolean()?,
            "intensity-target" => {
                cfg.intensity_target = if self.value == "auto" {
                    None
                } else {
                    Some(self.finite(f32::MIN_POSITIVE, 65504.0)?)
                }
            }
            "min-nits" => cfg.min_nits = self.finite(0.0, 65504.0)?,
            "relative-to-max-display" => cfg.relative_to_max_display = self.boolean()?,
            "linear-below" => cfg.linear_below = self.finite(0.0, 65504.0)?,
            _ => {
                return Err(format!(
                    "unknown jxl override '{}'; use -e jxl --list-overrides",
                    self.key
                ));
            }
        }
        Ok(())
    }
}
