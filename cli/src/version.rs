use crate::Encoder;

include!(concat!(env!("OUT_DIR"), "/versions.rs"));

pub fn encoder(encoder: Encoder) -> &'static str {
    match encoder {
        Encoder::Av1 => AV1,
        Encoder::Av2 => AV2,
        Encoder::JpegXl => JXL,
        Encoder::Jpeg => JPEG,
        Encoder::Hevc => HEVC,
        Encoder::Vvc => VVC,
    }
}

pub fn all() -> String {
    format!("{CLI}\n{AV1}\n{AV2}\n{JXL}\n{JPEG}\n{HEVC}\n{VVC}")
}
