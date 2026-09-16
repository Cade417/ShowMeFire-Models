import unittest

from risk_fusion.county_geometry import assign_regions, burnable_area_km2, load_counties


class CountyGeometryTests(unittest.TestCase):
    def setUp(self):
        self.counties = load_counties()

    def test_loads_all_115_missouri_counties(self):
        self.assertEqual(len(self.counties), 115)

    def test_fips_are_unique_and_five_digit_missouri_codes(self):
        fips_values = [c["fips"] for c in self.counties]
        self.assertEqual(len(fips_values), len(set(fips_values)))
        for fips in fips_values:
            self.assertEqual(len(fips), 5)
            self.assertTrue(fips.startswith("29"))

    def test_boone_county_area_is_plausible(self):
        boone = next(c for c in self.counties if c["fips"] == "29019")
        # Boone County, MO is ~1,772 km^2 (679 sq mi). Allow slack for the
        # county-boundary generalization in the source shapefile.
        self.assertGreater(boone["area_km2"], 1500)
        self.assertLess(boone["area_km2"], 2100)

    def test_burnable_area_falls_back_to_geometric_area_with_documented_source(self):
        county = self.counties[0]
        area, source = burnable_area_km2(county)
        self.assertEqual(area, county["area_km2"])
        self.assertEqual(source, "geometric_area_only")

    def test_region_assignment_covers_every_county_with_no_gaps(self):
        regions = assign_regions(self.counties)
        self.assertEqual(set(regions.keys()), {c["fips"] for c in self.counties})
        self.assertTrue(all(isinstance(v, int) for v in regions.values()))

    def test_region_assignment_is_deterministic(self):
        first = assign_regions(self.counties)
        second = assign_regions(self.counties)
        self.assertEqual(first, second)

    def test_every_region_has_at_least_one_county(self):
        regions = assign_regions(self.counties)
        from risk_fusion.county_geometry import N_LON_BINS, N_LAT_BINS
        counts = {}
        for region_id in regions.values():
            counts[region_id] = counts.get(region_id, 0) + 1
        self.assertEqual(len(counts), N_LON_BINS * N_LAT_BINS)
        self.assertTrue(all(count > 0 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
