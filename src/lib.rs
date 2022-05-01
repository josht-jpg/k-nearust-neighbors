use std::{
    collections::HashMap,
    ops::{Add, Sub},
};

#[derive(Debug, Clone)]
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

    #[test]
    fn linear_alg() {
        let v = vec![1., 5., -3.];
        let w = vec![0.5, 2., 3.];

        assert_eq!(v.dot(&w), 1.5);
        assert_eq!(v.subtract(&w), vec![0.5, 3., -6.]);
        assert_eq!(v.sum_of_squares(), 35.);
        assert_eq!(v.distance(&w), 45.25f64.sqrt());
    }

    macro_rules! await_fn {
        ($arg:expr) => {{
            tokio_test::block_on($arg)
        }};
    }

    async fn get_iris_data() -> Result<String, reqwest::Error> {
        let body = reqwest::get(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        )
        .await?
        .text()
        .await?;

        Ok(body)
    }

    type GenericResult<T> = Result<T, Box<dyn std::error::Error>>;

    fn process_iris_data<'a>(body: &str) -> GenericResult<Vec<LabeledPoint>> {
        body.split("\n")
            .filter(|data_point| data_point.len() > 0)
            .map(|data_point| -> GenericResult<LabeledPoint> {
                let columns = data_point.split(",").collect::<Vec<&str>>();
                let (label, point) = columns.split_last().ok_or("Cannot split last")?;
                let point = point
                    .iter()
                    .map(|feature| feature.parse::<f64>())
                    .collect::<Result<Vec<f64>, std::num::ParseFloatError>>()?;

                Ok(LabeledPoint { label, point })
            })
            .collect::<GenericResult<Vec<LabeledPoint>>>()
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

    #[test]
    fn iris() -> GenericResult<()> {
        let raw_iris_data = await_fn!(get_iris_data())?;
        let iris_data = process_iris_data(&raw_iris_data)?;

        let (train_set, test_set) = split_data(&iris_data, 0.70);
        assert_eq!(train_set.len(), 105);
        assert_eq!(test_set.len(), 45);

        let k = 5;
        let num_correct = count_correct_classifications(&train_set, &test_set, k);
        let percent_corrent = num_correct as f32 / test_set.len() as f32;

        assert!(percent_corrent > 0.9);

        Ok(())
    }
}
