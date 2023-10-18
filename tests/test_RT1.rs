#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

use std::collections::HashMap;

use RayBNN_Cell;

use RayBNN_Sparse::Util::Convert::get_global_weight_idx;


const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;

const TWO_F64: f64 = 2.0;

#[test]
fn test_RT1() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    arrayfire::set_seed(1232);

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<f32>();


	let neuron_size: u64 = 51000;
	let input_size: u64 = 4;
	let output_size: u64 = 3;
	let proc_num: u64 = 3;
	let active_size: u64 = 500000;
	let space_dims: u64 = 3;
	let mut batch_size: u64 = 105;

	let neuron_rad = 0.1;
    let time_step = 0.3;
    let nratio =  0.5;
    let neuron_std =  0.3;
    let sphere_rad =  4.0;


    let ray_input_connection_num = 1000000;

    let mut modeldata_float: HashMap<String, f64> = HashMap::new();
    let mut modeldata_int: HashMap<String, u64>  = HashMap::new();

    modeldata_int.insert("neuron_size".to_string(), neuron_size.clone());
    modeldata_int.insert("input_size".to_string(), input_size.clone());
    modeldata_int.insert("output_size".to_string(), output_size.clone());
    modeldata_int.insert("proc_num".to_string(), proc_num.clone());
    modeldata_int.insert("active_size".to_string(), active_size.clone());
    modeldata_int.insert("space_dims".to_string(), space_dims.clone());
    modeldata_int.insert("batch_size".to_string(), batch_size.clone());
    modeldata_int.insert("ray_input_connection_num".to_string(), ray_input_connection_num);
    modeldata_int.insert("ray_max_rounds".to_string(), 1000);
    modeldata_int.insert("ray_glia_intersect".to_string(), 1);
    modeldata_int.insert("ray_neuron_intersect".to_string(), 1);
    modeldata_int.insert("ray_gen_num_limit".to_string(), 10000);





    modeldata_float.insert("neuron_rad".to_string(), neuron_rad.clone());
    modeldata_float.insert("time_step".to_string(), time_step.clone());
    modeldata_float.insert("nratio".to_string(), nratio.clone());
    modeldata_float.insert("neuron_std".to_string(), neuron_std.clone());
    modeldata_float.insert("sphere_rad".to_string(), sphere_rad.clone());
    modeldata_float.insert("con_rad".to_string(), 40.0*neuron_rad.clone());



	let temp_dims = arrayfire::Dim4::new(&[4,1,1,1]);

	let mut glia_pos = arrayfire::constant::<f32>(0.0,temp_dims);
	let mut neuron_pos = arrayfire::constant::<f32>(0.0,temp_dims);


    let mut cell_pos: arrayfire::Array<f32>  = RayBNN_Cell::Hidden::Sphere::generate_uniform_sphere_posiiton(&modeldata_float, &modeldata_int);


    println!("cell_pos {}", cell_pos.dims()[0]);

    assert_eq!(cell_pos.dims()[0], active_size*2);
    assert_eq!(cell_pos.dims()[1], space_dims);

    let idx = RayBNN_Cell::Hidden::Sphere::check_cell_collision_minibatch(
        &modeldata_float, 
        &cell_pos
    );

    let idx = arrayfire::locate(&idx);

	cell_pos = arrayfire::lookup(&cell_pos, &idx, 0);


    RayBNN_Cell::Hidden::Sphere::split_into_glia_neuron(
        &modeldata_float,
    
        &cell_pos,
    
        &mut glia_pos,
        &mut neuron_pos
    );

    let temp_dims = arrayfire::Dim4::new(&[1,1,1,1]);
    let mut WRowIdxCOO = arrayfire::constant::<i32>(0,temp_dims);
    let mut WColIdx = arrayfire::constant::<i32>(0,temp_dims);


	let gen_dims = arrayfire::Dim4::new(&[neuron_pos.dims()[0],1,1,1]);
	let rep_dims = arrayfire::Dim4::new(&[1,1,1,1]);

    let mut input_idx_total = arrayfire::iota::<i32>(gen_dims,rep_dims);
    let hidden_idx_total = input_idx_total.clone();
    let input_pos_total = neuron_pos.clone();
    let hidden_pos_total = neuron_pos.clone();

    RayBNN_Raytrace::Tracing::RT1::RT1_random_rays(
        &modeldata_float,
        &modeldata_int,
    
    
        &input_pos_total,
        &input_idx_total,
    
    
        
        &mut WRowIdxCOO,
        &mut WColIdx
    );

    println!("WRowIdxCOO {}", WRowIdxCOO.dims()[0]);

    assert_eq!(WRowIdxCOO.dims()[0],WColIdx.dims()[0]  );
    assert!(WRowIdxCOO.dims()[0] >=  ray_input_connection_num );

    let gidx1 = get_global_weight_idx(
        neuron_size,
        &WRowIdxCOO,
        &WColIdx,
    );

    let unique = arrayfire::set_unique(&gidx1, false);
    assert_eq!(gidx1.dims()[0],unique.dims()[0]);


    let (max_val,_) = arrayfire::max_all(&WRowIdxCOO);
    assert!(max_val <= (neuron_pos.dims()[0] as i32));

    let (max_val,_) = arrayfire::max_all(&WColIdx);
    assert!(max_val <= (neuron_pos.dims()[0] as i32));

}
