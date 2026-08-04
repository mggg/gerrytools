use super::*;

#[test]
fn reports_sparse_districts_in_order() {
    let (slots, observed) = observed_districts(&[3, 1, 3]);
    assert_eq!(slots, 4);
    assert_eq!(district_ids(&observed), vec![1, 3]);
}

#[test]
#[should_panic(expected = "occupancy underflow")]
fn apply_panics_instead_of_wrapping_when_a_district_underflows() {
    let mut occupancy = DistrictOccupancy::new();
    occupancy.reset(&[1]);
    occupancy.apply(2, 1);
}

#[test]
fn tracks_districts_across_dynamic_words() {
    let (slots, observed) = observed_districts(&[0, 127, 128, 499]);
    assert_eq!(slots, 500);
    assert_eq!(district_ids(&observed), vec![0, 127, 128, 499]);

    let (slots, observed) = observed_districts(&[u16::MAX]);
    assert_eq!(slots, 1 << 16);
    assert_eq!(district_ids(&observed), vec![u16::MAX]);

    let mut occupancy = DistrictOccupancy::new();
    occupancy.reset(&[0, 127]);
    occupancy.apply(127, 499);
    assert_eq!(district_ids(occupancy.observed()), vec![0, 499]);
}
