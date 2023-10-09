#![allow(unused_parens)]
#![allow(non_snake_case)]

use arrayfire;
use RayBNN_DataLoader;

const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;


use rayon::prelude::*;



#[test]
fn test_rays_neurons_to_neurons() {
    arrayfire::set_backend(BACK_END);
    arrayfire::set_device(DEVICE);

    let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let ZERO = arrayfire::constant::<f64>(0.0, single_dims);


    //Generate rays starting from input neurons
    let mut start_line = ZERO.clone();
    let mut dir_line = ZERO.clone();

    

    let tile_dims = arrayfire::Dim4::new(&[hidden_size,1,1,1]);

    let mut tiled_input_idx =  arrayfire::tile(&input_idx, tile_dims);
    
    let mut tiled_hidden_idx = hidden_idx.clone();

    RayBNN_Raytrace::Generate::Fixed::tileDown(
        input_idx_size,
    
        &mut tiled_hidden_idx
    );

    RayBNN_Raytrace::Generate::Fixed::rays_from_neuronsA_to_neuronsB(
        con_rad,

        &input_pos,
        &hidden_pos,
    
        &mut start_line,
        &mut dir_line,

        &mut tiled_input_idx,
        &mut tiled_hidden_idx,
    );
    



}
