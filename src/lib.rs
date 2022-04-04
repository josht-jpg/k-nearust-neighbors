use std::{
    collections::HashMap,
    ops::{Add, Sub},
};

#[derive(Clone)]
struct LabeledPoint<'a> {
    point: Vec<f64>,
    label: &'a str,
}

trait LinearAlg<T>
where
    T: Add + Sub,
{
    fn dot(&self, w: &[T]) -> T;

    fn subtract(&self, w: &[T]) -> Vec<T>;

    fn sum_of_squares(&self) -> T;

    fn distance(&self, w: &[T]) -> f64;
}

impl LinearAlg<f64> for Vec<f64> {
    fn dot(&self, w: &[f64]) -> f64 {
        assert_eq!(self.len(), w.len());
        self.iter().zip(w).map(|(v_i, w_i)| v_i * w_i).sum()
    }

    fn subtract(&self, w: &[f64]) -> Vec<f64> {
        assert_eq!(self.len(), w.len());
        self.iter().zip(w).map(|(v_i, w_i)| v_i - w_i).collect()
    }

    fn sum_of_squares(&self) -> f64 {
        self.dot(&self)
    }

    fn distance(&self, w: &[f64]) -> f64 {
        assert_eq!(self.len(), w.len());
        self.subtract(w).sum_of_squares().sqrt()
    }
}

fn knn_classify(k: u8, data_points: &[LabeledPoint], new_point: &[f64]) -> Option<String> {
    let mut data_points_copy = data_points.to_vec();

    data_points_copy.sort_unstable_by(|a, b| {
        let dist_a = a.point.distance(new_point);
        let dist_b = b.point.distance(new_point);

        dist_a
            .partial_cmp(&dist_b)
            .expect("Cannot compare floating point numbers, encoutered a NAN")
    });

    let k_nearest_labels = &data_points_copy[..(k as usize)]
        .iter()
        .map(|a| a.label)
        .collect::<Vec<&str>>();

    let predicted_label = find_most_common_label(&k_nearest_labels);
    predicted_label
}

fn find_most_common_label(labels: &[&str]) -> Option<String> {
    let mut label_counts: HashMap<&str, u32> = HashMap::new();

    for label in labels.iter() {
        let current_label_count = if let Some(current_label_count) = label_counts.get(label) {
            *current_label_count
        } else {
            0
        };
        label_counts.insert(label, current_label_count + 1);
    }

    let most_common = label_counts
        .iter()
        .max_by(|(_label_a, count_a), (_label_b, count_b)| count_a.cmp(&count_b));

    if let Some((most_common_label, most_common_label_count)) = most_common {
        let is_tie_for_most_common = label_counts
            .iter()
            .any(|(label, count)| count == most_common_label_count && label != most_common_label);

        if !is_tie_for_most_common {
            return Some((*most_common_label).to_string());
        } else {
            let (_last, labels) = labels.split_last()?;
            return find_most_common_label(&labels);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{seq::SliceRandom, thread_rng};
    use rustlearn::datasets::iris;

    #[test]
    fn linear_alg() {
        let v = vec![1., 5., -3.];
        let w = vec![0.5, 2., 3.];

        assert_eq!(v.dot(&w), 1.5);
        assert_eq!(v.subtract(&w), vec![0.5, 3., -6.]);
        assert_eq!(v.sum_of_squares(), 35.);
        assert_eq!(v.distance(&w), 45.25f64.sqrt())
    }

    #[test]
    fn iris() {
        let (train_set, test_set) = split_data(&parse_iris_data(), 0.70);
        assert_eq!(train_set.len(), 105);
        assert_eq!(test_set.len(), 45);

        let k = 5;
        let num_correct = count_correct_classifications(&train_set, &test_set, k);
        let percent_corrent = num_correct as f32 / test_set.len() as f32;

        assert!(percent_corrent > 0.9)
    }

    fn split_data<T>(data: &[T], prob: f64) -> (Vec<T>, Vec<T>)
    where
        T: Clone,
    {
        let mut data_copy = data.to_vec();
        data_copy.shuffle(&mut thread_rng());
        let split_index = ((data.len() as f64) * prob).round() as usize;

        (
            data_copy[..split_index].to_vec(),
            data_copy[split_index..].to_vec(),
        )
    }

    fn count_correct_classifications(
        train_set: &[LabeledPoint],
        test_set: &[LabeledPoint],
        k: u8,
    ) -> u32 {
        let mut num_correct: u32 = 0;

        for iris in test_set.iter() {
            let predicted = knn_classify(k, &train_set, &iris.point);
            let actual = iris.label;

            if let Some(predicted) = predicted {
                if predicted == actual {
                    num_correct += 1;
                }
            }
        }

        num_correct
    }

    fn parse_iris_data<'a>() -> Vec<LabeledPoint<'a>> {
        let (measurements, labels) = iris::load_data();
        let measurements = measurements.data();
        let numeric_labels = labels.data();

        #[allow(non_snake_case)]
        let COLUMNS_PER_ROW = 4;

        (0..measurements.len())
            .step_by(COLUMNS_PER_ROW)
            .map(|i| LabeledPoint {
                point: vec![
                    measurements[i] as f64,
                    measurements[i + 1] as f64,
                    measurements[i + 2] as f64,
                    measurements[i + 3] as f64,
                ],
                label: get_label(Label::from(numeric_labels[i / 4])),
            })
            .collect::<Vec<LabeledPoint>>()
    }

    enum Label {
        Setosa = 0,
        Versicolor = 1,
        Virginica = 2,
    }

    impl From<f32> for Label {
        #[allow(illegal_floating_point_literal_pattern)]
        fn from(f: f32) -> Self {
            match f {
                0. => Self::Setosa,
                1. => Self::Versicolor,
                2. => Self::Virginica,
                _ => panic!("Labels have been incorrectly loaded"),
            }
        }
    }

    fn get_label<'a>(numeric_label: Label) -> &'a str {
        match numeric_label {
            Label::Setosa => "Setosa",
            Label::Versicolor => "Versicolor",
            Label::Virginica => "Virginica",
        }
    }
}
