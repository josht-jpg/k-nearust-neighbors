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

    fn distance(&self, w: &[f64]) -> T;
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

    return (data_copy[..cut].to_vec(), data_copy[cut..].to_vec());
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlearn::datasets::iris;

    #[test]
    fn iris() {
        let (measurements, labels) = iris::load_data();
        let measurements_data = measurements.data();
        let labels_data = labels.data();

        let labels: HashMap<u8, &str> =
            HashMap::from([(0, "Setosa"), (1, "Versicolor"), (2, "Virginica")]);

        let data = (0..measurements_data.len())
            .step_by(4)
            .map(|i| DataPoint {
                point: vec![
                    measurements_data[i] as f64,
                    measurements_data[i + 1] as f64,
                    measurements_data[i + 2] as f64,
                    measurements_data[i + 3] as f64,
                ],
                label: labels.get(&(labels_data[i / 4] as u8)).unwrap(),
            })
            .collect::<Vec<DataPoint>>();

        let (iris_train, iris_test) = split_data(&data, 0.70);

        assert_eq!(iris_train.len(), 105);
        assert_eq!(iris_test.len(), 45);

        let mut num_correct = 0;

        let k: u8 = 5;

        for iris in iris_test.iter() {
            let predicted = knn_classify(k, &iris_train, &iris.point);
            let actual = iris.label;

            if let Some(predicted) = predicted {
                if predicted == actual {
                    num_correct += 1;
                }
            }
        }

        assert!(num_correct as f32 / iris_test.len() as f32 > 0.9)
    }
}
