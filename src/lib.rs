use std::{cmp::Ordering, collections::HashMap};

fn max_key_value<K, V>(vote_counts: &HashMap<K, V>) -> Option<(&K, &V)>
where
    V: Ord,
{
    vote_counts
        .iter()
        .max_by(|a, b| a.1.cmp(&b.1))
        .map(|(k, v)| (k, v))
}

fn handle_label(vote_counts: &HashMap<&str, u32>, label: &str) -> u32 {
    if vote_counts.contains_key(label) {
        vote_counts.get(label).unwrap() + 1
    } else {
        1
    }
}

fn majority_vote(labels: &[&str]) -> Option<String> {
    let mut vote_counts: HashMap<&str, u32> = HashMap::new();

    for label in labels.iter() {
        vote_counts.insert(label, handle_label(&vote_counts, label));
    }

    if let Some((winner, winner_count)) = max_key_value(&vote_counts) {
        let num_winners = vote_counts.values().filter(|v| *v == winner_count).count();

        if num_winners == 1 {
            return Some((**winner).to_string());
        } else {
            return majority_vote(&labels[..labels.len() - 2]);
        }
    }

    None
}

#[derive(Clone, Debug)]
struct DataPoint<'a> {
    point: Vec<f64>,
    label: &'a str,
}

trait LinearAlg<T> {
    fn dot(&self, w: &[T]) -> T;

    fn subtract(&self, w: &[T]) -> Vec<T>;

    fn sum_of_squares(&self) -> T;

    fn squared_distance(&self, w: &[T]) -> T;

    fn distance(&self, w: &[T]) -> f64;
}

impl LinearAlg<f64> for Vec<f64> {
    fn dot(&self, w: &[f64]) -> f64 {
        self.iter().zip(w).map(|(v_i, w_i)| v_i * w_i).sum()
    }

    fn subtract(&self, w: &[f64]) -> Vec<f64> {
        assert_eq!(self.len(), w.len());
        self.iter().zip(w).map(|(v_i, w_i)| v_i - w_i).collect()
    }

    fn sum_of_squares(&self) -> f64 {
        self.dot(&self)
    }

    fn squared_distance(&self, w: &[f64]) -> f64 {
        self.subtract(w).sum_of_squares()
    }

    fn distance(&self, w: &[f64]) -> f64 {
        self.squared_distance(w).sqrt()
    }
}
impl LinearAlg<f32> for Vec<f32> {
    fn dot(&self, w: &[f32]) -> f32 {
        self.iter().zip(w).map(|(v_i, w_i)| v_i * w_i).sum()
    }

    fn subtract(&self, w: &[f32]) -> Vec<f32> {
        assert_eq!(self.len(), w.len());
        self.iter().zip(w).map(|(v_i, w_i)| v_i - w_i).collect()
    }

    fn sum_of_squares(&self) -> f32 {
        self.dot(&self)
    }

    fn squared_distance(&self, w: &[f32]) -> f32 {
        self.subtract(w).sum_of_squares()
    }

    fn distance(&self, w: &[f32]) -> f64 {
        self.squared_distance(w).sqrt() as f64
    }
}
impl LinearAlg<i32> for Vec<i32> {
    fn dot(&self, w: &[i32]) -> i32 {
        self.iter().zip(w).map(|(v_i, w_i)| v_i * w_i).sum()
    }

    fn subtract(&self, w: &[i32]) -> Vec<i32> {
        assert_eq!(self.len(), w.len());
        self.iter().zip(w).map(|(v_i, w_i)| v_i - w_i).collect()
    }

    fn sum_of_squares(&self) -> i32 {
        self.dot(&self)
    }

    fn squared_distance(&self, w: &[i32]) -> i32 {
        self.subtract(w).sum_of_squares()
    }

    fn distance(&self, w: &[i32]) -> f64 {
        (self.squared_distance(w) as f64).sqrt()
    }
}

fn knn_classify(k: u8, data_points: &[DataPoint], new_point: &[f64]) -> Option<String> {
    let mut data_copy = data_points.to_vec();

    data_copy.sort_by(|a, b| {
        let dist_a = a.point.distance(new_point);
        let dist_b = b.point.distance(new_point);

        if dist_a > dist_b {
            Ordering::Greater
        } else if dist_a == dist_b {
            Ordering::Equal
        } else {
            Ordering::Less
        }
    });

    let k_nearest_labels = &data_copy[..(k as usize)]
        .iter()
        .map(|a| a.label)
        .collect::<Vec<&str>>();

    majority_vote(&k_nearest_labels)
}

use rand::{seq::SliceRandom, thread_rng};

pub fn split_data<T>(data: &[T], prob: f64) -> (Vec<T>, Vec<T>)
where
    T: Clone,
{
    let mut data_copy = data.to_vec();
    data_copy.shuffle(&mut thread_rng());
    let cut = ((data.len() as f64) * prob).round() as usize;

    (data_copy[..cut].to_vec(), data_copy[cut..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlearn::datasets::iris;

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

    fn parse_iris_data<'a>() -> Vec<DataPoint<'a>> {
        let (measurements, labels) = iris::load_data();
        let measurements = measurements.data();
        let numeric_labels = labels.data();

        #[allow(non_snake_case)]
        let COLUMNS_PER_ROW = 4;

        (0..measurements.len())
            .step_by(COLUMNS_PER_ROW)
            .map(|i| DataPoint {
                point: vec![
                    measurements[i] as f64,
                    measurements[i + 1] as f64,
                    measurements[i + 2] as f64,
                    measurements[i + 3] as f64,
                ],
                label: get_label(Label::from(numeric_labels[i / 4])),
            })
            .collect::<Vec<DataPoint>>()
    }

    fn count_correct_classifications(
        train_set: &[DataPoint],
        test_set: &[DataPoint],
        k: u8,
    ) -> i32 {
        let mut num_correct = 0;

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
}
