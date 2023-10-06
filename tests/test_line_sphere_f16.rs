#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_line_sphere_f16() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);







    let mut start_line_cpu:Vec<f64> = vec![ 0.0, 0.0,      0.0, 0.0,          0.0, 0.0,         0.0, 0.0,       0.0, 0.0,            0.0, 0.0,         0.0, 0.0,                0.0, 0.0];
    let mut start_line = arrayfire::Array::new(&start_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    start_line = arrayfire::transpose(&start_line, false);
    
    
    
    
    //                                    All no hit 
    let mut dir_line_cpu:Vec<f64> = vec![ 0.0001, 0.0001,        0.0001, 0.0001,         0.0001, 0.0001,        0.0001, 0.0001,            0.0001, 0.0001,        0.0001, 0.0001,         0.0001, 0.0001,        0.0001, 0.0001,];
    let mut dir_line = arrayfire::Array::new(&dir_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    dir_line = arrayfire::transpose(&dir_line, false);
    
    
    
    
    
    let mut circle_center_cpu:Vec<f64> = vec![ 4.0, 1.0,      -4.0, -1.0,          7.0, -1.0       ];
    let mut circle_center = arrayfire::Array::new(&circle_center_cpu, arrayfire::Dim4::new(&[2, 3, 1, 1]));
    
    
    circle_center = arrayfire::transpose(&circle_center, false);
    
    
    
    
    
    let mut circle_radius_cpu:Vec<f64> = vec![ 0.5,   0.5,   1.0    ];
    let mut circle_radius = arrayfire::Array::new(&circle_radius_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));
    
    
    
    
    
    
    
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut intersect = arrayfire::constant::<bool>(false,single_dims);
    
    

    RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect(
        &start_line,
        &dir_line,
    
        &circle_center,
        &circle_radius,
    
        &mut intersect
        );
    
        assert_eq!(intersect.dims()[0], 8 );
        assert_eq!(intersect.dims()[1], 1 );
        assert_eq!(intersect.dims()[2], 3 );
    
    
    let mut intersect_cpu = vec!(bool::default();intersect.elements());
    intersect.host(&mut intersect_cpu);
    
    
    
    let intersect_act:Vec<bool> = vec![false, false, false, false, false, false, false, false                  ,false, false, false, false, false, false, false, false               ,false, false, false, false, false, false, false, false];
    
    assert_eq!(intersect_cpu, intersect_act);
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    let mut start_line_cpu:Vec<f64> = vec![ 0.0, 0.0,      0.0, 0.0,          0.0, 0.0,         0.0, 0.0,       0.0, 0.0,            0.0, 0.0,         0.0, 0.0,                0.0, 0.0];
    let mut start_line = arrayfire::Array::new(&start_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    start_line = arrayfire::transpose(&start_line, false);
    
    
    
    
    //                                    no hit         circle0            no hit               circle0          no hit               circle2           no hit                no hit
    let mut dir_line_cpu:Vec<f64> = vec![ 6.0, 3.0,      8.0, 3.0,          3.37, 1.011,         4.0, 1.0,        5.0, 0.5,            8.0, 0.0,         6.0, -0.5,             7.0, -2.2];
    let mut dir_line = arrayfire::Array::new(&dir_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    dir_line = arrayfire::transpose(&dir_line, false);
    
    
    
    
    
    let mut circle_center_cpu:Vec<f64> = vec![ 4.0, 1.0,      -4.0, -1.0,          7.0, -1.0       ];
    let mut circle_center = arrayfire::Array::new(&circle_center_cpu, arrayfire::Dim4::new(&[2, 3, 1, 1]));
    
    
    circle_center = arrayfire::transpose(&circle_center, false);
    
    
    
    
    
    let mut circle_radius_cpu:Vec<f64> = vec![ 0.5,   0.5,   1.0    ];
    let mut circle_radius = arrayfire::Array::new(&circle_radius_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));
    
    
    
    
    
    
    
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut intersect = arrayfire::constant::<bool>(false,single_dims);
    
    RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect(
        &start_line,
        &dir_line,
    
        &circle_center,
        &circle_radius,
    
        &mut intersect
        );
    
        assert_eq!(intersect.dims()[0], 8 );
        assert_eq!(intersect.dims()[1], 1 );
        assert_eq!(intersect.dims()[2], 3 );
    
    
    let mut intersect_cpu = vec!(bool::default();intersect.elements());
    intersect.host(&mut intersect_cpu);
    
    
    
    let intersect_act:Vec<bool> = vec![false, true, false, true, false, false, false, false               ,false, false, false, false, false, false, false, false              ,false, false, false, false, false, true, false, false];
    
    assert_eq!(intersect_cpu, intersect_act);
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    let mut start_line_cpu:Vec<f64> = vec![ 0.0, 0.0,      0.0, 0.0,          0.0, 0.0,         0.0, 0.0,       0.0, 0.0,            0.0, 0.0,         0.0, 0.0,                0.0, 0.0];
    let mut start_line = arrayfire::Array::new(&start_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    start_line = arrayfire::transpose(&start_line, false);
    
    
    
    
    //                                    no hit           circle1              no hit                 circle1            no hit               circle2           no hit                no hit
    let mut dir_line_cpu:Vec<f64> = vec![ -6.0, -3.0,      -8.0, -3.0,          -3.37, -1.011,         -4.0, -1.0,        -5.0, -0.5,            8.0, 0.0,         6.0, -0.5,             7.0, -2.2];
    let mut dir_line = arrayfire::Array::new(&dir_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    dir_line = arrayfire::transpose(&dir_line, false);
    
    
    
    
    
    let mut circle_center_cpu:Vec<f64> = vec![ 4.0, 1.0,      -4.0, -1.0,          7.0, -1.0       ];
    let mut circle_center = arrayfire::Array::new(&circle_center_cpu, arrayfire::Dim4::new(&[2, 3, 1, 1]));
    
    
    circle_center = arrayfire::transpose(&circle_center, false);
    
    
    
    
    
    let mut circle_radius_cpu:Vec<f64> = vec![ 0.5,   0.5,   1.0    ];
    let mut circle_radius = arrayfire::Array::new(&circle_radius_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));
    
    
    
    
    
    
    
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut intersect = arrayfire::constant::<bool>(false,single_dims);
    
    RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect(
        &start_line,
        &dir_line,
    
        &circle_center,
        &circle_radius,
    
        &mut intersect
        );
    
        assert_eq!(intersect.dims()[0], 8 );
        assert_eq!(intersect.dims()[1], 1 );
        assert_eq!(intersect.dims()[2], 3 );
    
    
    let mut intersect_cpu = vec!(bool::default();intersect.elements());
    intersect.host(&mut intersect_cpu);
    
    
    
    let intersect_act:Vec<bool> = vec![false, false, false, false, false, false, false, false               ,false, true, false, true, false, false, false, false             ,false, false, false, false, false, true, false, false];
    
    assert_eq!(intersect_cpu, intersect_act);
        
    //arrayfire::print_gen("intersect".to_string(), &intersect,Some(6));
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    let mut start_line_cpu:Vec<f64> = vec![ 5.0, 10.0,        10.0, -8.0,           -3.0, 1.0,          -2.0, -3.2,              -3.4, 1.2,              3.712, -1.312,                 -6.3, -2.41,                   -4.5, 1.25];
    let mut start_line = arrayfire::Array::new(&start_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    start_line = arrayfire::transpose(&start_line, false);
    
    
    
    
    //                                     circle1            no hit                circle3             circle2                   no hit                 circle0                         circle0,circle3               circle1,circle2
    let mut dir_line_cpu:Vec<f64> = vec![  -10.0, -10.0,      -15.0, 10.5,          -3.0, -3.0,         6.5, 1.95,                1.3, 0.5,              -4.233, 4.233,                  6.8, 4.76,                    9.3, -3.5];
    let mut dir_line = arrayfire::Array::new(&dir_line_cpu, arrayfire::Dim4::new(&[2, 8, 1, 1]));
    
    dir_line = arrayfire::transpose(&dir_line, false);
    
    
    
    
    
    
    let mut circle_center_cpu:Vec<f64> = vec![ -1.0, 2.0,        -4.0, 1.0,          4.0, -2.0,          -6.0, -2.0];
    let mut circle_center = arrayfire::Array::new(&circle_center_cpu, arrayfire::Dim4::new(&[2, 4, 1, 1]));
    
    
    circle_center = arrayfire::transpose(&circle_center, false);
    
    
    
    
    
    let mut circle_radius_cpu:Vec<f64> = vec![ 1.0,   0.5,   0.7 ,   0.2   ];
    let mut circle_radius = arrayfire::Array::new(&circle_radius_cpu, arrayfire::Dim4::new(&[4, 1, 1, 1]));
    
    
    
    
    
    
    
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut intersect = arrayfire::constant::<bool>(false,single_dims);
    
    RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect(
        &start_line,
        &dir_line,
    
        &circle_center,
        &circle_radius,
    
        &mut intersect
        );
    
    
        assert_eq!(intersect.dims()[0], 8 );
        assert_eq!(intersect.dims()[1], 1 );
        assert_eq!(intersect.dims()[2], 4 );
    
    //arrayfire::print_gen("intersect".to_string(), &intersect,Some(6));
    
    let mut intersect_cpu = vec!(bool::default();intersect.elements());
    intersect.host(&mut intersect_cpu);
    
    
    let intersect_act:Vec<bool> = vec![false, false, false, false, false, true, true, false,                    true, false, false, false, false, false, false, true,                false, false, false, true, false, false, false, true,               false, false, true, false, false, false, true, false];
    
    assert_eq!(intersect_cpu, intersect_act);
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    assert_eq!(intersect.dims()[0], 8 );
    assert_eq!(intersect.dims()[1], 1 );
    assert_eq!(intersect.dims()[2], 12 );

    //arrayfire::print_gen("intersect".to_string(), &intersect,Some(6));
    
    let mut intersect_cpu = vec!(bool::default();intersect.elements());
    intersect.host(&mut intersect_cpu);
    
    
    let intersect_act:Vec<bool> = vec![false, false, false, false, false, true, true, false,                    true, false, false, false, false, false, false, true,                false, false, false, true, false, false, false, true,               false, false, true, false, false, false, true, false,
                                        false, false, false, false, false, true, true, false,                    true, false, false, false, false, false, false, true,                false, false, false, true, false, false, false, true,               false, false, true, false, false, false, true, false,
                                        false, false, false, false, false, true, true, false,                    true, false, false, false, false, false, false, true,                false, false, false, true, false, false, false, true,               false, false, true, false, false, false, true, false];
    
    assert_eq!(intersect_cpu, intersect_act);
    */
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    let mut start_line_cpu:Vec<f64> = vec![ 0.99, 7.4,            3.3, 8.48,              4.54, 7.7,                6.52, 5.83,               3.01, 2.8,              7.17, -0.95,           0.74, -4.19  ];
    let mut start_line = arrayfire::Array::new(&start_line_cpu, arrayfire::Dim4::new(&[2, 7, 1, 1]));
    
    start_line = arrayfire::transpose(&start_line, false);
    
    
    
    
    //                                      no hit                 circle0                no hit                    circle0                   circle1                 no hit                 circle2
    let mut dir_line_cpu:Vec<f64> = vec![   2.48, -1.94,           0.54, -3.14,           -2.15, -3.84,             -3.64, -1.37,             3.41, -1.08,            -4.35, 3.57,           0.81, 6.96     ];
    let mut dir_line = arrayfire::Array::new(&dir_line_cpu, arrayfire::Dim4::new(&[2, 7, 1, 1]));
    
    dir_line = arrayfire::transpose(&dir_line, false);
    
    
    
    
    
    let mut circle_center_cpu:Vec<f64> = vec![ 4.0, 5.0,       7.0, 2.0,     2.0, 3.0 ];
    let mut circle_center = arrayfire::Array::new(&circle_center_cpu, arrayfire::Dim4::new(&[2, 3, 1, 1]));
    
    
    circle_center = arrayfire::transpose(&circle_center, false);
    
    
    
    
    
    let mut circle_radius_cpu:Vec<f64> = vec![ 0.5,  1.0,  0.7  ];
    let mut circle_radius = arrayfire::Array::new(&circle_radius_cpu, arrayfire::Dim4::new(&[3, 1, 1, 1]));
    
    
    
    
    
    
    
    
    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut intersect = arrayfire::constant::<bool>(false,single_dims);
    
    RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect(
        &start_line,
        &dir_line,
    
        &circle_center,
        &circle_radius,
    
        &mut intersect
        );
    
    
        assert_eq!(intersect.dims()[0], 7 );
        assert_eq!(intersect.dims()[1], 1 );
        assert_eq!(intersect.dims()[2], 3 );
    
    //arrayfire::print_gen("intersect".to_string(), &intersect,Some(6));
    
    let mut intersect_cpu = vec!(bool::default();intersect.elements());
    intersect.host(&mut intersect_cpu);
    
    let intersect_act:Vec<bool> = vec![false, true, false, true, false, false, false,              false, false, false, false, true, false, false,                 false, false, false, false, false, false, true ];
    
    
    
    assert_eq!(intersect_cpu, intersect_act);
      

}
