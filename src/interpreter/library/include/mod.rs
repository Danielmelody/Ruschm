use super::LibraryName;

lazy_static! {
    // sorted
    pub static ref LIBRARY_STRS: Vec<(LibraryName, &'static str)> = {
        let mut v = vec![
            (
                library_name!("scheme", "base"),
                include_str!("scheme/base.sld"),
            ),
            (
                library_name!("scheme", "write"),
                include_str!("scheme/write.sld"),
            ),
        ];
        v.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        v
    };
}
