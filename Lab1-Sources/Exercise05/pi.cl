__kernel void get_pi(const long num_steps, 
const double step,
__global double* sum)
{
    // chaque thread calcule x et 4/(1+x*x)
    // reduction sur les thread pour calculer la somme
    int i = get_global_id(0);               
    if(i < num_steps)  {
        double tmp_x = (i-0.5)*step;
        sum += 4.0/(1.0+x*x);                 
    } 
}