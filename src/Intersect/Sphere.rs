use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;




const TWO_f64: f64 = 2.0;
const ONE_f64: f64 = 1.0;
const ZERO_f64: f64 = 0.0;

const EPSILON_f64: f64 = 1.0e-3;

const ONEMINUSEPSILON_f64: f64 = ONE_f64 - EPSILON_f64;




pub fn line_sphere_intersect<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
	start_line: &arrayfire::Array<Z>,
	dir_line: &arrayfire::Array<Z>,

	circle_center: &arrayfire::Array<Z>,
	circle_radius: &arrayfire::Array<Z>,

	intersect: &mut arrayfire::Array<bool>
	)
{
	let line_num: u64 = start_line.dims()[0];

	let circle_num: u64 = circle_center.dims()[0];

	let space_dims = start_line.dims()[1];


	let t_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let temp_constant = vec![TWO_f64 ];
	let mut TWO = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ONE_f64 ];
	let mut ONE = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();

	let temp_constant = vec![ZERO_f64 ];
	let mut ZERO = arrayfire::Array::new(&temp_constant, t_dims).cast::<Z>();




	// C^T
	let mut CENTERSUBSTART = arrayfire::reorder_v2(&circle_center, 2, 1, Some(vec![0]));


	// C - S
	CENTERSUBSTART = arrayfire::sub(&CENTERSUBSTART,start_line,true);


	// dot(C - S, D)
	let mut dotret = arrayfire::mul(&CENTERSUBSTART,dir_line,true);

	dotret = arrayfire::sum(&dotret,1);



	// |D|^2
	let mut sq = arrayfire::pow(dir_line,&TWO,false);
	sq = arrayfire::sum(&sq, 1);




	// dot(C - S, D)  /  |D|^2
	dotret = arrayfire::div(&dotret,&sq,true);
	drop(sq);

	// Clamp(     dot(C - S, D)  /  |D|^2      )
	dotret = arrayfire::clamp(&dotret, &ZERO, &ONE, false);


	// Clamp(     dot(C - S, D)  /  |D|^2      )   D
	dotret = arrayfire::mul( &dotret,dir_line, true);



	// (C - S)   -   Clamp( dot(C - S, D)  /  |D|^2  ) D
    dotret = CENTERSUBSTART - dotret;


	// Mag( Vector Rejection )
	dotret = arrayfire::pow(&dotret,&TWO,false);
	dotret = arrayfire::sum(&dotret, 1);


	// R^T
	let mut tempradius = arrayfire::reorder_v2(&circle_radius, 2, 1, Some(vec![0]));

	// R^2
	tempradius = arrayfire::pow(&tempradius,&TWO,false);

	//  (tempradius >= tempdir )
	*intersect = arrayfire::ge(&tempradius, &dotret, true);



}





pub fn line_sphere_intersect_batch(
	batch_size: u64,
	start_line: &arrayfire::Array<f64>,
	dir_line: &arrayfire::Array<f64>,

	circle_center: &arrayfire::Array<f64>,
	circle_radius: &arrayfire::Array<f64>,

	intersect: &mut arrayfire::Array<bool>
    )
{

    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = circle_radius.dims()[0];





    startseq = i;
    endseq = i + batch_size-1;
    if (endseq >= (total_size-1))
    {
        endseq = total_size-1;
    }

    let seqs = &[arrayfire::Seq::new(startseq as f64, endseq as f64, 1.0 as f64)];
    let input_circle_radius  = arrayfire::index(circle_radius, seqs);

    let seqs2 = &[arrayfire::Seq::new(startseq as f64, endseq as f64, 1.0 as f64), arrayfire::Seq::default()];
    let input_circle_center  = arrayfire::index(circle_center, seqs2);


	line_sphere_intersect(
		start_line,
		dir_line,
	
		&input_circle_center,
		&input_circle_radius,
	
		intersect
		);


    i = i + batch_size;


	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut intersect_temp = arrayfire::constant::<bool>(false,single_dims);


    while i < total_size
    {
        startseq = i;
        endseq = i + batch_size-1;
        if (endseq >= (total_size-1))
        {
            endseq = total_size-1;
        }

		let seqs = &[arrayfire::Seq::new(startseq as f64, endseq as f64, 1.0 as f64)];
		let input_circle_radius  = arrayfire::index(circle_radius, seqs);
	
		let seqs2 = &[arrayfire::Seq::new(startseq as f64, endseq as f64, 1.0 as f64), arrayfire::Seq::default()];
		let input_circle_center  = arrayfire::index(circle_center, seqs2);
	
		line_sphere_intersect(
			start_line,
			dir_line,
		
			&input_circle_center,
			&input_circle_radius,
		
			&mut intersect_temp
			);
        

		*intersect = arrayfire::join(2, intersect, &intersect_temp);

        i = i + batch_size;
    }


}




