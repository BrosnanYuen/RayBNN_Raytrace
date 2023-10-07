#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_generate() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);


    let con_rad = 8.2;

    let target_input_pos_cpu: Vec<f32> = vec![4.1, 1.7, -0.9];
	let mut target_input_pos = arrayfire::Array::new(&target_input_pos_cpu, arrayfire::Dim4::new(&[1, 3, 1, 1]));

    //arrayfire::print_gen("target_input_pos".to_string(), &target_input_pos, Some(6));

    let input_pos_cpu: Vec<f32> = vec![4.1, 1.7, -0.9,
                                        0.3, -2.0, 5.0,
                                        -1.0, 0.2, -3.1,
                                        9.0, -4.0, -6.2,
                                        0.3, 9.9, -5.1,
                                        -7.2, 0.6, -3.8,
                                        3.4, 2.0, 2.7];
	let mut input_pos = arrayfire::Array::new(&input_pos_cpu, arrayfire::Dim4::new(&[3, 7, 1, 1]));
    input_pos = arrayfire::transpose(&input_pos, false);

    //arrayfire::print_gen("input_pos".to_string(), &input_pos, Some(6));


    let input_idx_cpu: Vec<i32> = vec![1, 4, 6, 8, 12, 14, 15];
	let mut input_idx = arrayfire::Array::new(&input_idx_cpu, arrayfire::Dim4::new(&[7, 1, 1, 1]));


    RayBNN_Raytrace::Generate::Ray::filter_rays(
        con_rad,
    
        &target_input_pos,
    
        &mut input_pos,
        &mut input_idx,
    );
    //arrayfire::print_gen("input_pos".to_string(), &input_pos, Some(6));

    let mut input_pos_act_cpu: Vec<f32> = vec![4.1, 1.7, -0.9,
                                        0.3, -2.0, 5.0,
                                        -1.0, 0.2, -3.1,
                                        3.4, 2.0, 2.7];
	let mut input_pos_act = arrayfire::Array::new(&input_pos_act_cpu, arrayfire::Dim4::new(&[3, 4, 1, 1]));
    input_pos_act = arrayfire::transpose(&input_pos_act, false);

    input_pos_act.host(&mut input_pos_act_cpu);

    let mut input_pos_pred = vec!(f32::default();input_pos.elements());
	input_pos.host(&mut input_pos_pred);


    assert_eq!(input_pos_pred, input_pos_act_cpu);



    let mut input_idx_pred = vec!(i32::default();input_idx.elements());
	input_idx.host(&mut input_idx_pred);

    let mut input_idx_act: Vec<i32> = vec![1, 4, 6, 15];

    assert_eq!(input_idx_pred, input_idx_act);




}
