use crate::{Args, Chroma, EncodingEffort, is_gray};
use image::DynamicImage;
use jhm17::{
    ChromaSubsamplingMethod, Deadzone, PixelDensity, Polish, PolishOptions, ProgressiveMode,
    QuantizationTableType, RdoOptions, SamplePrecision, SamplingFactor, StepTuning,
};

#[derive(Debug, Clone)]
pub(crate) struct JpegOptions {
    pub quality: u8,
    pub sampling: SamplingFactor,
    pub subsampling: ChromaSubsamplingMethod,
    pub precision: SamplePrecision,
    pub progressive: Option<ProgressiveMode>,
    pub restart_interval: u16,
    pub optimize_huffman: bool,
    pub arithmetic: bool,
    pub rdo_enabled: bool,
    pub rdo: RdoOptions,
    pub luma_table: QuantizationTableType,
    pub chroma_table: QuantizationTableType,
    pub luma_steps: StepTuning,
    pub cb_steps: StepTuning,
    pub cr_steps: StepTuning,
    pub deadzone: Deadzone,
    pub auto_chroma_scale: bool,
    pub polish: Polish,
    pub density: PixelDensity,
}

impl Default for JpegOptions {
    fn default() -> Self {
        Self {
            quality: 80,
            sampling: SamplingFactor::R_4_2_0,
            subsampling: ChromaSubsamplingMethod::Average,
            precision: SamplePrecision::All,
            progressive: None,
            restart_interval: 0,
            optimize_huffman: true,
            arithmetic: false,
            rdo_enabled: true,
            rdo: RdoOptions::default(),
            luma_table: QuantizationTableType::ImprovedDetectionModel,
            chroma_table: QuantizationTableType::ImprovedDetectionModel,
            luma_steps: StepTuning {
                bands: jhm17::DEFAULT_LUMA_BANDS,
                ..StepTuning::default()
            },
            cb_steps: StepTuning::default(),
            cr_steps: StepTuning::default(),
            deadzone: Deadzone::default(),
            auto_chroma_scale: true,
            polish: Polish::TrellisGreedy,
            density: PixelDensity::default(),
        }
    }
}

impl JpegOptions {
    pub fn from_args(args: &Args) -> Result<Self, String> {
        let mut options = Self {
            quality: args.quality,
            sampling: match args.chroma.unwrap_or(Chroma::C420) {
                Chroma::C444 => SamplingFactor::R_4_4_4,
                Chroma::C422 => SamplingFactor::R_4_2_2,
                Chroma::C420 => SamplingFactor::R_4_2_0,
            },
            polish: Self::polish_for_speed(args.speed),
            ..Self::default()
        };
        for option in &args.overrides {
            option.apply_jpeg(&mut options)?;
        }
        options.validate()?;
        Ok(options)
    }

    pub fn polish_for_speed(speed: EncodingEffort) -> Polish {
        match speed {
            EncodingEffort::Fast => Polish::Greedy,
            EncodingEffort::Medium => Polish::Trellis,
            EncodingEffort::Slow | EncodingEffort::ExtraSlow => Polish::TrellisGreedy,
        }
    }

    pub fn polish_options(&self) -> PolishOptions {
        self.polish
            .options()
            .unwrap_or_else(|| Polish::TrellisGreedy.options().unwrap())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.rdo_enabled && self.rdo.aq_min > self.rdo.aq_max {
            return Err("JPEG rdo-aq-min must not exceed rdo-aq-max".into());
        }
        if self.polish != Polish::Off
            && !matches!(
                self.sampling,
                SamplingFactor::F_1_1
                    | SamplingFactor::R_4_4_4
                    | SamplingFactor::F_2_2
                    | SamplingFactor::R_4_2_0
            )
        {
            return Err("jhm17 polishing requires chroma 420 or 444; use jpeg:polish=off for other sampling modes".into());
        }
        Ok(())
    }

    pub fn encoder<W: jhm17::JfifWrite>(&self, writer: W) -> jhm17::Encoder<W> {
        let mut encoder = jhm17::Encoder::new(writer, self.quality);
        encoder.set_sampling_factor(self.sampling);
        encoder.set_chroma_subsampling_method(self.subsampling);
        encoder.set_sample_precision(self.precision);
        encoder.set_progressive_mode(self.progressive);
        encoder.set_restart_interval(self.restart_interval);
        encoder.set_optimized_huffman_tables(self.optimize_huffman);
        encoder.set_arithmetic_coding(self.arithmetic);
        encoder.set_rdo(self.rdo_enabled.then_some(self.rdo));
        encoder.set_quantization_tables(self.luma_table.clone(), self.chroma_table.clone());
        encoder.set_step_tuning(self.luma_steps, self.cb_steps);
        encoder.set_chroma_step_tuning(self.cb_steps, self.cr_steps);
        encoder.set_deadzone(self.deadzone);
        encoder.set_auto_chroma_scale(self.auto_chroma_scale);
        encoder.set_polish(self.polish);
        encoder.set_density(self.density);
        encoder
    }
}

pub(crate) fn encode_jpeg(
    img: &DynamicImage,
    args: &Args,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> anyhow::Result<Vec<u8>> {
    let options = JpegOptions::from_args(args).map_err(anyhow::Error::msg)?;
    let width = u16::try_from(img.width())
        .map_err(|_| anyhow::anyhow!("JPEG width exceeds 65535 pixels"))?;
    let height = u16::try_from(img.height())
        .map_err(|_| anyhow::anyhow!("JPEG height exceeds 65535 pixels"))?;
    let mut output = Vec::new();
    let mut encoder = options.encoder(&mut output);
    if let Some(icc) = icc {
        encoder.add_icc_profile(icc)?;
    }
    if let Some(exif) = exif {
        encoder.add_exif_metadata(exif.strip_prefix(b"Exif\0\0").unwrap_or(exif))?;
    }
    let stats = std::sync::Arc::new(std::sync::Mutex::new(None));
    if args.verbose {
        encoder.set_polish_stats_sink(stats.clone());
        eprintln!(
            "jpeg   : jhm17 quality={} sampling={:?} polish={:?}",
            options.quality, options.sampling, options.polish
        );
    }
    if is_gray(img.color()) {
        if args.verbose && options.polish != Polish::Off {
            eprintln!(
                "jpeg   : jhm17 polishing applies to RGB input; grayscale uses the configured quantizer"
            );
        }
        encoder.encode(
            img.to_luma8().as_raw(),
            width,
            height,
            jhm17::ColorType::Luma,
        )?;
    } else {
        // JPEG has no alpha; RGB also enables jhm17's decoder-feedback polish.
        encoder.encode(img.to_rgb8().as_raw(), width, height, jhm17::ColorType::Rgb)?;
    }
    if args.verbose
        && let Some(stats) = stats.lock().unwrap().as_ref()
    {
        eprintln!(
            "polish : {} evaluated, {} accepted, {} estimated bits",
            stats.evaluated, stats.accepted, stats.bits_delta
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use img_parts::{ImageEXIF, ImageICC};

    fn image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(64, 48, |x, y| {
            image::Rgb([
                ((x * 17 + y * 11) % 256) as u8,
                ((y * 23 + x * 3) % 256) as u8,
                ((x * y + y * 7) % 256) as u8,
            ])
        }))
    }

    #[test]
    fn default_polishing_really_executes_on_rgb_pixels() {
        let image = image();
        let args = crate::args::try_parse(&["cli", "-e", "jpeg", "-q", "60", "in.png"]).unwrap();
        let options = JpegOptions::from_args(&args).unwrap();
        let stats = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut bytes = Vec::new();
        let mut encoder = options.encoder(&mut bytes);
        encoder.set_polish_stats_sink(stats.clone());
        encoder
            .encode(image.to_rgb8().as_raw(), 64, 48, jhm17::ColorType::Rgb)
            .unwrap();
        let stats = stats
            .lock()
            .unwrap()
            .expect("default JPEG mode must run the polishing pass");
        assert!(stats.evaluated > 0);
        let decoded = image::load_from_memory(&bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (64, 48));
    }

    #[test]
    fn jpeg_raster_output_uses_jhm17_and_the_grouped_overrides() {
        let image = image();
        let args = crate::args::try_parse(&[
            "cli",
            "-q",
            "99",
            "-s",
            "slow",
            "in.png",
            "out.jpg",
            "-O",
            "jpeg:quality=60,chroma=444,progressive=standard,polish=greedy,restart-interval=3",
        ])
        .unwrap();
        let actual = crate::raster::encode_raster(
            &image,
            crate::raster::RasterFormat::Jpeg,
            &args,
            None,
            None,
            None,
        )
        .unwrap();
        let mut expected = Vec::new();
        let mut encoder = jhm17::Encoder::new(&mut expected, 60);
        encoder.set_sampling_factor(SamplingFactor::R_4_4_4);
        encoder.set_progressive_mode(Some(ProgressiveMode::Standard));
        encoder.set_polish(Polish::Greedy);
        encoder.set_restart_interval(3);
        encoder
            .encode(image.to_rgb8().as_raw(), 64, 48, jhm17::ColorType::Rgb)
            .unwrap();
        assert_eq!(actual, expected);
        let decoded = image::load_from_memory(&actual).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (64, 48));
    }

    #[test]
    fn jpeg_preserves_icc_and_exif_without_duplicate_exif_prefixes() {
        let image = image();
        let args =
            crate::args::try_parse(&["cli", "-e", "jpeg", "-O", "jpeg:polish=off", "in.png"])
                .unwrap();
        let icc = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let tiff = b"II\x2a\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        for exif in [tiff.to_vec(), [b"Exif\0\0".as_slice(), tiff].concat()] {
            let encoded = encode_jpeg(&image, &args, Some(&icc), Some(&exif)).unwrap();
            let jpeg = img_parts::jpeg::Jpeg::from_bytes(encoded.into()).unwrap();
            assert_eq!(jpeg.icc_profile().unwrap().as_ref(), icc.as_slice());
            assert_eq!(jpeg.exif().unwrap().as_ref(), tiff);
        }
    }

    #[test]
    fn jpeg_keeps_grayscale_and_discards_unsupported_alpha() {
        let image = DynamicImage::ImageLumaA8(image::ImageBuffer::from_fn(9, 7, |x, y| {
            image::LumaA([(x * 23 + y * 7) as u8, 127])
        }));
        let args = crate::args::try_parse(&["cli", "-e", "jpeg", "in.png"]).unwrap();
        let encoded = encode_jpeg(&image, &args, None, None).unwrap();
        let decoded = image::load_from_memory(&encoded).unwrap();
        assert_eq!(decoded.color(), image::ColorType::L8);
        assert_eq!((decoded.width(), decoded.height()), (9, 7));
    }

    #[test]
    fn tuning_lists_and_custom_polish_options_are_forwarded() {
        let args = crate::args::try_parse(&["cli", "-e", "jhm17", "in.png", "-O", "jpg:polish=off,polish-judge=ssimulacra2,polish-rounds=3,polish-trellis-k=4,rdo-lambda=0.08,rdo-aq-min=0.25,rdo-aq-max=1.5,luma-table=flat,chroma-bands=[1,2,3,4,5],density=[300,240],density-unit=dpi,sample-precision=luma,cr-ac=1.5,chroma-dc=0.75"]).unwrap();
        let options = JpegOptions::from_args(&args).unwrap();
        let polish = options.polish.options().unwrap();
        assert_eq!(polish.judge, jhm17::PolishJudge::Ssimulacra2);
        assert_eq!(polish.rounds, 3);
        assert_eq!(polish.trellis_k, 4);
        assert_eq!(options.rdo.lambda, 0.08);
        assert_eq!((options.rdo.aq_min, options.rdo.aq_max), (0.25, 1.5));
        assert_eq!(options.cb_steps.bands, [1., 2., 3., 4., 5.]);
        assert_eq!(options.cr_steps.bands, options.cb_steps.bands);
        assert_eq!(options.cr_steps.ac, 1.5);
        assert_eq!(options.cb_steps.ac, 1.0);
        assert_eq!((options.cb_steps.dc, options.cr_steps.dc), (0.75, 0.75));
        let encoder = options.encoder(Vec::new());
        assert!(matches!(
            encoder.quantization_tables()[0],
            QuantizationTableType::Flat
        ));
        assert_eq!(encoder.sample_precision(), SamplePrecision::Luma);
        assert_eq!(
            encoder.density(),
            PixelDensity {
                density: (300, 240),
                unit: jhm17::PixelDensityUnit::Inches
            }
        );
    }
}
