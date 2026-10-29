use super::EncoderOverride;

impl EncoderOverride {
    pub fn apply_heic(&self, cfg: &mut hpvca::EncodeConfig) -> Result<(), String> {
        if self.pipeline_setting()?.is_some() {
            return Ok(());
        }
        match self.key.as_str() {
            "quality" => cfg.quality = self.number(1, 100, "1..100")?,
            "threads" => cfg.threads = self.threads()?,
            "speed" => {
                cfg.speed = match self.value.as_str() {
                    "fast" => hpvca::Speed::Fast,
                    "slow" => hpvca::Speed::Slow,
                    _ => return Err(self.invalid("fast|slow")),
                }
            }
            "parallelism" => {
                cfg.parallelism = match self.value.as_str() {
                    "single" => hpvca::ParallelismStrategy::Single,
                    "wpp" => hpvca::ParallelismStrategy::Wpp,
                    "tiles-wpp" | "tileswpp" => hpvca::ParallelismStrategy::TilesWpp,
                    "grid" => hpvca::ParallelismStrategy::Grid,
                    "grid-wpp" | "gridwpp" => hpvca::ParallelismStrategy::GridWpp,
                    _ => return Err(self.invalid("single|wpp|tiles-wpp|grid|grid-wpp")),
                }
            }
            "sao" => cfg.sao = self.boolean()?,
            "screen-content" => cfg.screen_content = self.boolean()?,
            "implicit-rdpcm" => cfg.implicit_rdpcm = self.boolean()?,
            "persistent-rice" => cfg.persistent_rice = self.boolean()?,
            "lossless-ycbcr" => cfg.lossless_ycbcr = self.boolean()?,
            "orientation" => {
                cfg.metadata.orientation = hpvca::Orientation::from_exif(self.orientation()? as u16)
            }
            "max-cll" => {
                cfg.metadata
                    .content_light_level
                    .get_or_insert(hpvca::ContentLightLevel::new(0, 0))
                    .max_content_light_level = self.number(0, u16::MAX, "0..65535 nits")?
            }
            "max-fall" => {
                cfg.metadata
                    .content_light_level
                    .get_or_insert(hpvca::ContentLightLevel::new(0, 0))
                    .max_pic_average_light_level = self.number(0, u16::MAX, "0..65535 nits")?
            }
            "color-encoding" => {
                let current = cfg.color.color_encoding();
                let mut color = crate::heic::hpvca_cicp(self.source_color()?);
                color.matrix = current.matrix;
                color.full_range = current.full_range;
                cfg.color.cicp = Some(color);
            }
            _ => {
                return Err(format!(
                    "unknown heic override '{}'; use -e heic --list-overrides",
                    self.key
                ));
            }
        }
        Ok(())
    }
}
