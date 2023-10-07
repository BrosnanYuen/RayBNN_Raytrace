#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_line_sphere_batchV2() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);





    let mut start_line_cpu:Vec<f64> = vec![ 5.0, 10.0,        10.0, -8.0,           -3.0, 1.0,          -2.0, -3.2,              -3.4, 1.2,              3.712, -1.312,                 -6.3, -2.41,                   -4.5, 1.25];
    let mut start_line = arrayfire::Array::new(&start_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    start_line = arrayfire::transpose(&start_line, false);
    
    
    
    
    //                                     circle1            no hit                circle3             circle2                   no hit                 circle0                         circle0,circle3               circle1,circle2
    let mut dir_line_cpu:Vec<f64> = vec![  -10.0, -10.0,      -15.0, 10.5,          -3.0, -3.0,         6.5, 1.95,                1.3, 0.5,              -4.233, 4.233,                  6.8, 4.76,                    9.3, -3.5];
    let mut dir_line = arrayfire::Array::new(&dir_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    dir_line = arrayfire::transpose(&dir_line, false);
    
    
    
    
    
    
    let mut circle_center_cpu:Vec<f64> = vec![ -1.0, 2.0,        -4.0, 1.0,          4.0, -2.0,          -6.0, -2.0,              -1.0, 2.0,        -4.0, 1.0,          4.0, -2.0,          -6.0, -2.0,            -1.0, 2.0,        -4.0, 1.0,          4.0, -2.0,          -6.0, -2.0];
    let mut circle_center = arrayfire::Array::new(&circle_center_cpu, arrayfire::Dim4::new(&[2, 12, 1, 1]));
    
    
    circle_center = arrayfire::transpose(&circle_center, false);
    
    
    
    
    
    let mut circle_radius_cpu:Vec<f64> = vec![1.0,   0.5,   0.7 ,   0.2,                1.0,   0.5,   0.7 ,   0.2,                1.0,   0.5,   0.7 ,   0.2];
    let mut circle_radius = arrayfire::Array::new(&circle_radius_cpu, arrayfire::Dim4::new(&[12, 1, 1, 1]));
    
    
    
    
    
    
    
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut intersect = arrayfire::constant::<bool>(false,single_dims);
    
    /*
    RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect_batch(
        3,
        &start_line,
        &dir_line,
    
        &circle_center,
        &circle_radius,
    
        &mut intersect
    );
    */
    let input_idx_cpu: Vec<i32> = vec![12, 42, 61, 84,     126, 142, 151, 210];
	let mut input_idx = arrayfire::Array::new(&input_idx_cpu, arrayfire::Dim4::new(&[8, 1, 1, 1]));

    let hidden_idx_cpu: Vec<i32> = vec![912, 942, 961, 984,     9126, 9142, 9151, 9210];
	let mut hidden_idx = arrayfire::Array::new(&hidden_idx_cpu, arrayfire::Dim4::new(&[8, 1, 1, 1]));

    RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect_batchV2(
        3,
    
        1,
    
        &circle_center,
        &circle_radius,
    
        &mut start_line,
        &mut dir_line,
    
        &mut input_idx,
        &mut hidden_idx,
    );


    arrayfire::print_gen("input_idx".to_string(), &input_idx,Some(6));
    
    
    


}
