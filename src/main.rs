extern crate mnist;
extern crate nalgebra;
extern crate typenum;
extern crate futures;
extern crate futures_cpupool;
extern crate num_cpus;

use mnist::{Mnist, MnistBuilder};
use nalgebra::{Matrix, MatrixArray, U28, VectorN};
use typenum::{U785};

type Image = Matrix<f32, U28, U28, MatrixArray<f32, U28, U28>>;
type WeightVector = VectorN<f32, U785>;

const EPSILON: f32 = 0.01;
const LEARNING_STEP: f32 = 0.5;

fn flatten_image(image: &Image) -> WeightVector {
    let mut flatten_image = WeightVector::zeros();
    flatten_image[0] = 1.0;
    for i in 1..flatten_image.len() {
        flatten_image[i] = image[i - 1];
    }
    flatten_image
}

fn hypothesis(weigts: &WeightVector, image: &Image) -> f32 {
    hypothesis_flatten(&weigts, &flatten_image(&image))
}

fn hypothesis_flatten(coefficient: &WeightVector, flatten_image: &WeightVector) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-coefficient.dot(&flatten_image)))
}

fn print_i(image: &WeightVector) {
    for i in 1..image.len() {
        print!("{} ", format!("{:.*}", 2, image[i]));
        if i % 28 == 0 {
            println!();
        }
    }
    println!();
}

fn weight(examples: &Vec<(u32, WeightVector)>, number_learning: u32) -> WeightVector {
    let mut weights = WeightVector::zeros();
    let mut i: u32 = 0;
    let len = examples.len() as f32;
    loop {
        let mut max_step: f32 = 0.0;
        weights = {
            let iter = weights
                .iter()
                .enumerate()
                .map(|(j, w)| {
                    let step: f32 = examples
                        .iter()
                        .map(|&(actual_number, image)| {
                            let actual_result = if number_learning == actual_number {
                                1.0
                            } else {
                                0.0
                            };
                            (hypothesis_flatten(&weights, &image) - actual_result) * image[j]
                        })
                        .sum();
                    let step = step / len;
                    if step.abs() > max_step.abs() {
                        max_step = step;
                    }
                    w - LEARNING_STEP * step
                });
            WeightVector::from_iterator(iter)
        };

        i += 1;
        if max_step.abs() <= EPSILON || i > 10 {
            break;
        }
    }
    weights
}

fn main() {
    println!("Gathering training and testing sets...");

    let mut test_count: [u32; 10] = [0; 10];
    let (normalized_examples, normalized_tests) = {
        let mut normalized_examples: Vec<(u32, WeightVector)> = Vec::new();
        let mut normalized_tests: Vec<Vec<WeightVector>> = vec![Vec::new(); 10];

        let (trn_size, _rows, _cols, tst_size) = (100 as usize, 28, 28, 1_000 as usize);
        // Deconstruct the returned Mnist struct.
        let Mnist { trn_img, trn_lbl, val_img: _val_img, val_lbl: _val_lbl, tst_img, tst_lbl } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(trn_size as u32)
            .validation_set_length(0)
            .test_set_length(tst_size as u32)
            .finalize();

        for i in 0..trn_size {
            let label = trn_lbl[i] as u32;
            let iter = trn_img[i * 28 * 28..(i + 1) * 28 * 28].iter().map(|&x| x as f32 / 255.0);
            let image: Image = Image::from_iterator(iter);
            normalized_examples.push((label, flatten_image(&image)));
        }

        for i in 0..tst_size {
            let label = tst_lbl[i];
            let iter = tst_img[i * 28 * 28..(i + 1) * 28 * 28].iter().map(|&x| x as f32 / 255.0);
            let image: Image = Image::from_iterator(iter);
            normalized_tests[label as usize].push(flatten_image(&image));
            test_count[label as usize] += 1;
        }

        (normalized_examples, normalized_tests)
    };

    println!("Learning...");

    use futures::Future;
    use futures_cpupool::{CpuPool, CpuFuture};
    let pool = CpuPool::new(num_cpus::get());

    let weights: Vec<WeightVector> = {
        let futures: Vec<CpuFuture<WeightVector, std::io::Error>> = (0..10)
            .map(|i| {
                let examples = normalized_examples.clone();
                pool.spawn_fn(move || -> Result<WeightVector, std::io::Error> {
                    println!("Learning for {}", i);
                    Ok(weight(&examples, i as u32))
                })
            })
            .collect();

        futures::future::join_all(futures).wait().unwrap()
    };

    println!("Testing...");

    let mut hits: [u32; 10] = [0; 10];
    for (label, normalized_test_images) in normalized_tests.iter().enumerate() {
        for image in normalized_test_images.iter() {
            let result: usize = weights
                .iter().enumerate()
                .map(|(i, w)| (i, hypothesis_flatten(w, &image)))
                .fold((0, 0.0),|(max_i, max_result), (i, result)| {
                    if result > max_result {
                        (i, result)
                    } else {
                        (max_i, max_result)
                    }
                }).0;

            if result == label {
                hits[result] += 1;
            }
        }
    }
    for i in 0..10 {
        let hit_rate: f32 = hits[i] as f32 / test_count[i] as f32 * 100.0;
        println!("{}: {}/{} ({}%)", i, hits[i], test_count[i], hit_rate);
    }
}
