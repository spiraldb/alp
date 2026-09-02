//! Subsampling shared by the two encoders' parameter searches.
//!
//! Both searches look at a bounded number of values, since the property they are after — how
//! many decimal digits a column uses, or which leading bits its values share — belongs to the
//! column rather than to any one value. This module decides which values they look at.

use std::ops::Range;

/// Which elements of an input a parameter search examines.
///
/// A plan is a set of disjoint, ascending, contiguous ranges. It is built once and shared by every
/// candidate the search tries, so the trials all score the same values and their estimates stay
/// comparable.
#[derive(Debug)]
pub(crate) struct SamplePlan {
    ranges: Vec<Range<usize>>,
    count: usize,
}

impl SamplePlan {
    /// A plan covering every element of a `len`-element input.
    pub(crate) fn full(len: usize) -> Self {
        let mut ranges = Vec::new();
        if len > 0 {
            ranges.push(0..len);
        }
        Self { ranges, count: len }
    }

    /// A plan covering at most `max_sample` elements of a `len`-element input, as evenly spread
    /// contiguous runs of `block` values.
    ///
    /// Runs, rather than a fixed stride, are what make subsampling safe. A stride of
    /// `len / max_sample` aliases with any periodicity in the input — interleaved coordinate or
    /// embedding columns, round-robin sensor readings, a header value every so many rows — and a
    /// strided sample then observes only one phase of the data. A run sees every phase of any
    /// period up to its length, and touches far fewer cache lines than a wide stride.
    ///
    /// Falls back to [`Self::full`] for inputs already at or below `max_sample`.
    pub(crate) fn subsample(len: usize, max_sample: usize, block: usize) -> Self {
        let block = block.clamp(1, max_sample.max(1));
        if len <= max_sample {
            return Self::full(len);
        }

        let n_blocks = (max_sample / block).max(1);
        // `len > max_sample >= n_blocks * block` puts the spacing at `block` or more, so the runs
        // stay disjoint, and the last one starts at `len - block` or earlier, so all are in bounds.
        // Multiplying the floored spacing (rather than dividing a product) keeps this from
        // overflowing on absurd lengths.
        let spacing = if n_blocks > 1 {
            (len - block) / (n_blocks - 1)
        } else {
            0
        };
        let ranges: Vec<Range<usize>> = (0..n_blocks)
            .map(|i| {
                let start = i * spacing;
                start..start + block
            })
            .collect();

        let count = ranges.iter().map(Range::len).sum();
        Self { ranges, count }
    }

    /// The ranges to sample, ascending and disjoint.
    #[cfg(test)]
    pub(crate) fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    /// Total number of elements the plan visits.
    pub(crate) fn count(&self) -> usize {
        self.count
    }

    /// The elements of `values` the plan selects, in input order.
    ///
    /// # Panics
    ///
    /// Panics if the plan was built for a longer input than `values`.
    pub(crate) fn iter<'a, T>(&'a self, values: &'a [T]) -> impl Iterator<Item = &'a T> + 'a {
        self.ranges
            .iter()
            .flat_map(move |range| values[range.start..range.end].iter())
    }
}

#[cfg(test)]
mod test {
    use super::SamplePlan;

    const MAX_SAMPLE: usize = 4096;
    const BLOCK: usize = 64;

    #[test]
    fn test_sample_plan_covers_short_inputs_fully() {
        for len in [0usize, 1, 63, 64, 4095, MAX_SAMPLE] {
            let plan = SamplePlan::subsample(len, MAX_SAMPLE, BLOCK);
            assert_eq!(plan.count(), len, "short inputs must be scanned in full");
            let covered: usize = plan.ranges().iter().map(|r| r.len()).sum();
            assert_eq!(covered, len);
        }
    }

    #[test]
    fn test_sample_plan_ranges_are_in_bounds_and_disjoint() {
        // Includes lengths that are and are not multiples of the block and sample sizes, and a
        // budget too small to hold more than one block.
        for (max_sample, block) in [(MAX_SAMPLE, BLOCK), (64, 8), (32, 8), (8, 8), (7, 8)] {
            for len in [
                max_sample + 1,
                2 * max_sample,
                3 * max_sample + 1,
                100_003,
                1 << 20,
            ] {
                let plan = SamplePlan::subsample(len, max_sample, block);
                assert!(
                    plan.count() <= max_sample,
                    "subsampling must honour the budget of {max_sample} for len {len}"
                );
                assert_eq!(
                    plan.count(),
                    plan.ranges().iter().map(|r| r.len()).sum::<usize>()
                );

                let mut prev_end = 0;
                for range in plan.ranges() {
                    assert!(
                        range.start >= prev_end,
                        "ranges must be ascending, disjoint"
                    );
                    assert!(
                        range.end <= len,
                        "range {}..{} exceeds {len}",
                        range.start,
                        range.end
                    );
                    prev_end = range.end;
                }
                assert!(!plan.ranges().is_empty());
            }
        }
    }

    #[test]
    fn test_sample_plan_spans_the_input() {
        // The first run starts at the front and the last one ends at the back, so the sample spans
        // the column rather than its head.
        let len = 100_003;
        let plan = SamplePlan::subsample(len, 64, 8);
        assert_eq!(plan.ranges().len(), 8);
        assert_eq!(plan.ranges()[0].start, 0);
        let last = plan.ranges().last().expect("eight ranges");
        assert!(last.end > len - 8, "last run {last:?} should reach the end");
    }

    #[test]
    fn test_iter_visits_the_planned_elements_in_order() {
        let values: Vec<usize> = (0..1000).collect();
        let plan = SamplePlan::subsample(values.len(), 32, 8);
        let sampled: Vec<usize> = plan.iter(&values).copied().collect();
        assert_eq!(sampled.len(), plan.count());

        let expected: Vec<usize> = plan.ranges().iter().flat_map(Clone::clone).collect();
        assert_eq!(sampled, expected);

        let full: Vec<usize> = SamplePlan::full(values.len())
            .iter(&values)
            .copied()
            .collect();
        assert_eq!(full, values);
    }
}
