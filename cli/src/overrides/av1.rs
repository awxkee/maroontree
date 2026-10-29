use super::EncoderOverride;

impl EncoderOverride {
    pub fn apply_av1(&self, cfg: &mut maroontree::EncodeConfig) -> Result<(), String> {
        if self.pipeline_setting()?.is_some() {
            return Ok(());
        }
        match self.key.as_str() {
            "orientation" => {
                cfg.metadata.orientation =
                    maroontree::Orientation::from_exif(self.orientation()? as u16)
            }
            "max-cll" => {
                cfg.metadata
                    .content_light_level
                    .get_or_insert(maroontree::ContentLightLevel::new(0, 0))
                    .max_content_light_level = self.number(0, u16::MAX, "0..65535 nits")?
            }
            "max-fall" => {
                cfg.metadata
                    .content_light_level
                    .get_or_insert(maroontree::ContentLightLevel::new(0, 0))
                    .max_pic_average_light_level = self.number(0, u16::MAX, "0..65535 nits")?
            }
            "color-encoding" => {
                let current = cfg
                    .color_encoding
                    .unwrap_or_else(maroontree::Cicp::srgb_ycbcr);
                let color = self
                    .source_color()?
                    .to_maroontree(current.matrix, current.full_range);
                cfg.color_encoding = Some(maroontree::Cicp {
                    primaries: color.primaries,
                    transfer: color.transfer,
                    ..current
                });
            }
            "quality" => cfg.quality = self.number(1, 100, "1..100")?,
            "threads" => cfg.threads = self.threads()?,
            "speed" => {
                cfg.speed = match self.value.as_str() {
                    "extraslow" => maroontree::Speed::ExtraSlow,
                    "slow" => maroontree::Speed::Slow,
                    "medium" => maroontree::Speed::Medium,
                    "fast" => maroontree::Speed::Fast,
                    _ => return Err(self.invalid("extraslow|slow|medium|fast")),
                }
            }
            "updating-cdf" => cfg.updating_cdf = self.boolean()?,
            "adaptive-quant" => cfg.adaptive_quant = self.boolean()?,
            "variance-boost" => cfg.variance_boost = self.boolean()?,
            "dark-aq" => cfg.dark_aq = self.boolean()?,
            "screen-content" => cfg.screen_content = self.boolean()?,
            "intrabc" => cfg.intrabc = self.boolean()?,
            "cdef" => cfg.cdef = self.boolean()?,
            "wiener" => cfg.wiener = self.boolean()?,
            "quantization-matrices" => cfg.quantization_matrices = self.boolean()?,
            "qmatrix-level" => {
                cfg.qmatrix_level = if self.value == "auto" {
                    None
                } else {
                    Some(self.number(0, 15, "auto or 0..15")?)
                };
                cfg.quantization_matrices = true;
            }
            _ => {
                return Err(format!(
                    "unknown av1 override '{}'; use -e av1 --list-overrides",
                    self.key
                ));
            }
        }
        Ok(())
    }
}
