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
rayon = "1.7.0"
num = "0.4.1"
num-traits = "0.2.16"
half = { version = "2.3.1" , features = ["num-traits"] }
RayBNN_DataLoader = "0.1.3"
RayBNN_Sparse = "0.1.2"
RayBNN_Raytrace = "0.1.4"
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




