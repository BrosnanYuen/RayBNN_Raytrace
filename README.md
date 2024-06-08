# RayBNN_Raytrace

Ray tracing library using GPUs, CPUs, and FPGAs via CUDA, OpenCL, and oneAPI 


Raytraces intersections between rays, spheres, circles


# Install Arrayfire

Install the Arrayfire 3.9.0 binaries at [https://arrayfire.com/binaries/](https://arrayfire.com/binaries/)

or build from source
[https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire](https://github.com/arrayfire/arrayfire/wiki/Getting-ArrayFire)




# Add to Cargo.toml
```
arrayfire = { version = "3.8.1", package = "arrayfire_fork" }
rayon = "1.10.0"
num = "0.4.3"
num-traits = "0.2.19"
half = { version = "2.4.1" , features = ["num-traits"] }
RayBNN_DataLoader = "2.0.2"
RayBNN_Sparse = "2.0.1"
RayBNN_Cell = "2.0.2"
RayBNN_Raytrace = "2.0.2"
```



# List of Examples


# Line Sphere Intersection
```
RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect(
    &start_line,
    &dir_line,

    &circle_center,
    &circle_radius,

    &mut intersect
);
```

# Line Sphere Intersection Batch
```
RayBNN_Raytrace::Intersect::Sphere::line_sphere_intersect_batch(
    3,
    &start_line,
    &dir_line,

    &circle_center,
    &circle_radius,

    &mut intersect
);
```


# Line Sphere Intersection Batch V2
```
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
```


# Raytrace Neural Connections using RT3
```
RayBNN_Raytrace::Tracing::RT3::RT3_distance_limited_directly_connected(
    &modeldata_float,
    &modeldata_int,

    &glia_pos,

    &input_pos_total,
    &input_idx_total,

    &hidden_pos_total,
    &hidden_idx_total,

    
    &mut WRowIdxCOO,
    &mut WColIdx
);
```



