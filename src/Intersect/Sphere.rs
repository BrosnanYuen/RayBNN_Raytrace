use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;




const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;

const EPSILON_F64: f64 = 1.0e-3;

const ONEMINUSEPSILON_F64: f64 = ONE_F64 - EPSILON_F64;

const COUNT_LIMIT: u64 = 10000000000;


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

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();

	let ONE = arrayfire::constant::<f64>(ONE_F64,single_dims).cast::<Z>();




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





pub fn line_sphere_intersect_batch<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
	batch_size: u64,
	start_line: &arrayfire::Array<Z>,
	dir_line: &arrayfire::Array<Z>,

	circle_center: &arrayfire::Array<Z>,
	circle_radius: &arrayfire::Array<Z>,

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

	let input_circle_radius  = arrayfire::rows(circle_radius, startseq  as i64,endseq as i64);

	let input_circle_center  = arrayfire::rows(circle_center, startseq  as i64,endseq as i64);

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

		
		let input_circle_radius  = arrayfire::rows(circle_radius, startseq  as i64,endseq as i64);

		let input_circle_center  = arrayfire::rows(circle_center, startseq  as i64,endseq as i64);

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





pub fn line_sphere_intersect_batchV2<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
	batch_size: u64,

	threshold: u32,

	circle_center: &arrayfire::Array<Z>,
	circle_radius: &arrayfire::Array<Z>,

	start_line: &mut arrayfire::Array<Z>,
	dir_line: &mut arrayfire::Array<Z>,

	input_idx: &mut arrayfire::Array<i32>,
	hidden_idx: &mut arrayfire::Array<i32>,
)
{

    let mut i: u64 = 0;
    let mut startseq: u64 = 0;
    let mut endseq: u64 = 0;
    let total_size = circle_radius.dims()[0];


	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut intersect = arrayfire::constant::<bool>(false,single_dims);




	let ray_num = start_line.dims()[0];
	let counter_dims = arrayfire::Dim4::new(&[ray_num,1,1,1]);
	let mut counter = arrayfire::constant::<u32>(0,counter_dims);


    startseq = i;
    endseq = i + batch_size-1;
    if (endseq >= (total_size-1))
    {
        endseq = total_size-1;
    }



	let input_circle_radius  = arrayfire::rows(circle_radius, startseq  as i64,endseq as i64);

	let input_circle_center  = arrayfire::rows(circle_center, startseq  as i64,endseq as i64);

	line_sphere_intersect(
		start_line,
		dir_line,
	
		&input_circle_center,
		&input_circle_radius,
	
		&mut intersect
	);

	let mut counter_temp = intersect.cast::<u8>();
	
	counter = counter + arrayfire::sum(&counter_temp, 2);





    i = i + batch_size;


	let mut period = 1 + (COUNT_LIMIT/(intersect.elements() as u64));
	let mut clean_counter = 1;
	let mut CMPRET = arrayfire::constant::<bool>(false,single_dims);
	let mut idx_intersect = arrayfire::constant::<u32>(0,single_dims);


    while i < total_size
    {
        startseq = i;
        endseq = i + batch_size-1;
        if (endseq >= (total_size-1))
        {
            endseq = total_size-1;
        }



		let input_circle_radius  = arrayfire::rows(circle_radius, startseq  as i64,endseq as i64);

		let input_circle_center  = arrayfire::rows(circle_center, startseq  as i64,endseq as i64);
	

		line_sphere_intersect(
			start_line,
			dir_line,
		
			&input_circle_center,
			&input_circle_radius,
		
			&mut intersect
		);
        
		counter_temp = intersect.cast::<u8>();
		
		counter = counter + arrayfire::sum(&counter_temp, 2);
	

		clean_counter = clean_counter + 1;
		if clean_counter >= period
		{
			

			//  (threshold >= counter )
			CMPRET = arrayfire::ge(&threshold, &counter, false);
			//Lookup  1 >= dir_line  >= 0
			idx_intersect = arrayfire::locate(&CMPRET);

			counter = arrayfire::lookup(&counter, &idx_intersect, 0);
			*start_line = arrayfire::lookup(start_line, &idx_intersect, 0);
			*dir_line = arrayfire::lookup(dir_line, &idx_intersect, 0);
			*input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);
			*hidden_idx = arrayfire::lookup(hidden_idx, &idx_intersect, 0);
		
		
			period = 1 + (COUNT_LIMIT/(intersect.elements() as u64));
			clean_counter = 1;
			CMPRET = arrayfire::constant::<bool>(false,single_dims);
			idx_intersect = arrayfire::constant::<u32>(0,single_dims);

		}

		

        i = i + batch_size;
    }

	//  (threshold >= counter )
	CMPRET = arrayfire::ge(&threshold, &counter, false);
	//Lookup  1 >= dir_line  >= 0
	idx_intersect = arrayfire::locate(&CMPRET);

	
	*start_line = arrayfire::lookup(start_line, &idx_intersect, 0);
	*dir_line = arrayfire::lookup(dir_line, &idx_intersect, 0);
	*input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);
	*hidden_idx = arrayfire::lookup(hidden_idx, &idx_intersect, 0);

}




