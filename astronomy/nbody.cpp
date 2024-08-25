/*
Solves the nbody problem and outputs the positions of the moving objects at 
regular time intervals.

Uses individual time steps for each planet and a sixth-order Hermite scheme. 
The temporal loop uses adaptive timestepping.

The initial velocity and acceleration vectors are specified in main.

References:
Aarseth, S.J. (2003). "(Cambridge Monographs on Mathematical Physics) Gravitational 
    N-body Simulations: Tools and Algorithms". Cambridge University Press.
    ISBN: 978-0521121538
*/

#include <cfloat>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "omp.h"

// accuracy adjustment parameter
#define ETA 0.05f

// Use Aarseth's timesteps
#define USE_AARSETH_TIMESTEPS 1

#define MIN_TIMESTEP 1e-10f
//#define MIN_TIMESTEP DBL_MIN
#define TOL 1e-10f

#define NUM_PLANETS 3
#define PRINT_INTERVAL 0.01f
#define ENDTIME 500.0f

#define OPENMP 1

// Returns the size of the new timestep
double get_timestep_rel_error(double a_new[][3], double a[][3], int ptm, double t_old)
{
    double df[3] = {a_new[ptm][0]-a[ptm][0],a_new[ptm][1]-a[ptm][1],a_new[ptm][2]-a[ptm][2]};
    double numer = sqrt(a_new[ptm][0]*a_new[ptm][0]+a_new[ptm][1]*a_new[ptm][1]+a_new[ptm][2]*a_new[ptm][2]);
    double denom = sqrt(df[0]*df[0]+df[1]*df[1]+df[2]*df[2]);
    double t_new = ETA*t_old*pow(numer/denom,1.0/8.0);
    if(t_new < TOL)
    {
        return MIN_TIMESTEP;
    }
    else
    {
        return t_new;
    }
}

// Returns the size of the new timestep
double get_timestep_aarseth(double a[][3], double s[][3], double j[][3], double c[][3], int i, int id, double sv[][35])
{
    sv[id][31]=a[i][0]*a[i][0]+a[i][1]*a[i][1]+a[i][2]*a[i][2];
    sv[id][32]=s[i][0]*s[i][0]+s[i][1]*s[i][1]+s[i][2]*s[i][2];
    sv[id][34]=c[i][0]*c[i][0]+c[i][1]*c[i][1]+c[i][2]*c[i][2];
    if(abs(sv[id][33]*sv[id][34]+sv[id][32]*sv[id][32]) < TOL)
    {
        return MIN_TIMESTEP;
    }
    else
    {
        return ETA*sqrt((sv[id][31]*sv[id][32]+sv[id][33]*sv[id][33])/(sv[id][33]*sv[id][34]+sv[id][32]*sv[id][32]));
    }
}

// Updates the acceleration, jerk, snap and crackle for planet i
void get_new_ajsc_sixth(double r[][3],double v[][3],double a[][3],double j[][3], double s[][3], double c[][3],
                        double ai[][3], double ji[][3], double si[][3], double ci[][3],
                        int i, double m[NUM_PLANETS], double sv[][35], int id, int nt)
{
    for(int n = id; n < NUM_PLANETS; n=n+nt)
    {
        if(n == i)
        {
            continue;
        }
        
        sv[n][0] = r[n][0]-r[i][0];
        sv[n][1] = r[n][1]-r[i][1];
        sv[n][2] = r[n][2]-r[i][1];
        sv[n][3] = v[n][0]-v[i][0];
        sv[n][4] = v[n][1]-v[i][1];
        sv[n][5] = v[n][2]-v[i][2];
        sv[n][6] = a[n][0]-a[i][0];
        sv[n][7] = a[n][1]-a[i][1];
        sv[n][8] = a[n][2]-a[i][2];
        sv[n][9] = j[n][0]-j[i][0];
        sv[n][10] = j[n][1]-j[i][1];
        sv[n][11] = j[n][2]-j[i][2];
        sv[n][12] = sv[n][0]*sv[n][0]+sv[n][1]*sv[n][1]+sv[n][2]*sv[n][2];
        sv[n][13] = pow(sv[n][12],1.50);
        sv[n][14] = sv[n][3]*sv[n][3]+sv[n][4]*sv[n][4]+sv[n][5]*sv[n][5];
        sv[n][15] = sv[n][0]*sv[n][3]+sv[n][1]*sv[n][4]+sv[n][2]*sv[n][5];
        sv[n][16] = sv[n][0]*sv[n][6]+sv[n][1]*sv[n][7]+sv[n][2]*sv[n][8];
        sv[n][17] = sv[n][0]*sv[n][9]+sv[n][1]*sv[n][10]+sv[n][2]*sv[n][11];
        sv[n][18] = sv[n][3]*sv[n][6]+sv[n][4]*sv[n][7]+sv[n][5]*sv[n][8];
        sv[n][19] = sv[n][15] / sv[n][12];
        sv[n][20] = ((sv[n][14]+sv[n][16])/(sv[n][12]))+sv[n][19]*sv[n][19];
        sv[n][21] = ((3.0*sv[n][18]+sv[n][17])/(sv[n][12]))+sv[n][19]*(3.0*sv[n][20]-4.0*sv[n][19]*sv[n][19]);
        sv[n][22]=m[n]/sv[n][13];

        for(int k = 0; k < 3; k++)
        {
            a[i][k]=0.0;
            j[i][k]=0.0;
            s[i][k]=0.0;
            c[i][k]=0.0;
        }
        
        for(int k = 0; k < 3; k++)
        {
            a[i][k]+=sv[n][22]*sv[n][k];
        }
        for(int k = 0; k < 3; k++)
        {
            j[i][k]+=sv[n][22]*sv[n][k+3]-3.0*sv[n][19]*a[i][k];
        }
        for(int k = 0; k < 3; k++)
        {
            s[i][k]+=sv[n][22]*sv[n][k+6]-6.0*sv[n][19]*j[i][k]-3.0*sv[n][20]*a[i][k];
        }
        for(int k = 0; k < 3; k++)
        {
            c[i][k]+=sv[n][22]*sv[n][k+9]-9.0*sv[n][19]*s[i][k]-9.0*sv[n][20]*j[i][k]-3.0*sv[n][21]*a[i][k];
        }
    }
}


// Predict the future positions
void get_all_predicted_positions_sixth(double r[][3],double v[][3],double a[][3],double j[][3],double s[][3],double c[][3],
                        double r_new[][3],double v_new[][3],double a_new[][3],double j_new[][3], 
                        double m[NUM_PLANETS], double dt, double sv[][35], int id, int nt)
{
    // Maclaurin
    sv[id][23]=dt;
    sv[id][24]=pow(dt,2.0)/2.0;
    sv[id][25]=pow(dt,3.0)/6.0;
    sv[id][26]=pow(dt,4.0)/24.0;
    sv[id][27]=pow(dt,5.0)/120.0;

    for(int p = id; p < NUM_PLANETS; p=p+nt)
    {
        for(int i = 0; i < 3; i++)
        {
            r_new[p][i]=r[p][i]+sv[id][23]*v[p][i]+sv[id][24]*a[p][i]+sv[id][25]*j[p][i]+sv[id][26]*s[p][i]+sv[id][27]*c[p][i];
            v_new[p][i]=v[p][i]+sv[id][23]*a[p][i]+sv[id][24]*j[p][i]+sv[id][25]*s[p][i]+sv[id][26]*c[p][i];
            a_new[p][i]=a[p][i]+sv[id][23]*j[p][i]+sv[id][24]*s[p][i]+sv[id][25]*c[p][i];
            j_new[p][i]=j[p][i]+sv[id][23]*s[p][i]+sv[id][24]*c[p][i];
        }
    }
}

// Get the position correction 
void get_corrected_position_sixth(double r0[][3], double v0[][3], double a0[][3], double j0[][3], double s0[][3], double c0[][3],
                                  double r1[][3], double v1[][3], double a1[][3], double j1[][3], double s1[][3], double c1[][3],
                                  double dt, int i, double sv[][35], int id, int nt)
{
    sv[id][28]=dt/2.0;
    sv[id][29]=-pow(dt,2.0)/10.0;
    sv[id][30]=pow(dt,3.0)/120.0;

    if(id < 4)
    {
        for(int n = id; n < 3; n=n+nt)
        {
            r0[i][n]=r0[i][n]+sv[id][28]*(v1[i][n]+v0[i][n])+sv[id][29]*(a1[i][n]-a0[i][n])+sv[id][30]*(j1[i][n]+j0[i][n]);
            v0[i][n]=v0[i][n]+sv[id][28]*(a1[i][n]+a0[i][n])+sv[id][29]*(j1[i][n]-j0[i][n])+sv[id][30]*(s1[i][n]+s0[i][n]);
        }
    }
}

// Move a single planet
void single_step_sixth(double r[][3],double v[][3],double a[][3],double j[][3],double s[][3],double c[][3],
                        double r_new[][3],double v_new[][3],double a_new[][3],double j_new[][3],double s_new[][3],double c_new[][3],
                        double ai[][3], double ji[][3], double si[][3], double ci[][3],
                        double m[NUM_PLANETS], double * global_t, double times[NUM_PLANETS], double timesteps[NUM_PLANETS], double sv[][35],
                        int id, int nt)
{
    // select the planet to move
    int ptm = 0;
    double tnew = times[ptm]+timesteps[ptm];
    for(int i = 0; i < NUM_PLANETS; i++)
    {
        double t_tmp=times[i]+timesteps[i];
        if(t_tmp < tnew)
        {
            ptm = i;
        }
    }

    // Increment time
    *global_t=times[ptm]+timesteps[ptm];
    times[ptm]+=timesteps[ptm];

    // Get the predicted positions
    get_all_predicted_positions_sixth(r_new,v_new,a_new,j_new,r,v,a,j,s,c,m,timesteps[ptm],sv,id,nt);
    
    // Get new values for planet ptm
    get_new_ajsc_sixth(r,v,a,j,s,c,ai,ji,si,ci,ptm,m,sv,id,nt);

    // Get the new timestep
#ifdef USE_AARSETH_TIMESTEPS
    timesteps[ptm]=get_timestep_aarseth(a,s,j,c,ptm,id,sv);
#else
    timesteps[ptm]=get_timestep_rel_error(a_new,a,ptm,dt);
#endif    

#pragma omp barrier

    // Hermite interpolation
    get_corrected_position_sixth(r,v,a,j,s,c,r_new,v_new,a_new,j_new,s_new,c_new,timesteps[ptm],ptm,sv,id,nt);
}

// The temporal loop
void temporal_loop(double r[][3], double v[][3], double a[][3], double j[][3], double s[][3], double c[][3],
                double r_new[][3], double v_new[][3], double a_new[][3], double j_new[][3], double s_new[][3], double c_new[][3],
                double ai[][3], double ji[][3], double si[][3], double ci[][3],
                double m[NUM_PLANETS], double times[NUM_PLANETS], double timesteps[NUM_PLANETS], double sv[][35],
                int id, int nt)
{
    double global_time = 0;
    double * global_t = &global_time;
    double next_interval=0.0;

    if(id == 0)
    {
        std::cout << NUM_PLANETS << std::endl;
    }

    while(*global_t < ENDTIME)
    {
        // Write output to stdout
        if(*global_t > next_interval && id == 0)
        {
            std::cout << "t:" << *global_t << std::endl;
            for(int i = 0; i < NUM_PLANETS; i++)
            {
                std::cout << i << ":" << std::fixed << std::setprecision(8) << r[i][0] << "," << r[i][1] << "," << r[i][2] << std::endl;
            }
            next_interval+=PRINT_INTERVAL;
        }

        // Move an object
        single_step_sixth(r,v,a,j,s,c,r_new,v_new,a_new,j_new,s_new,c_new,ai,ji,si,ci,m,global_t,times,timesteps,sv,id,nt);        
    }
}

int main()
{
    // The position and velocity vectors at t=0
    double (*r)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0},{1.0,1.0,0.0},{0.0,-1.0,0.0}};
    double (*v)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0},{-1.0,0.0,0.0},{1.0,0.0,0.0}};

    // The mass of each planet
    double (*m) = new double [NUM_PLANETS] {5.0,1.0,1.0};
    
    // Declare a bunch of arrays
    double (*a)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    double (*j)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*s)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*c)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*ai)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*ji)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*si)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*ci)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*times) = new double[NUM_PLANETS] {{0.0}};
    double (*timesteps) = new double[NUM_PLANETS] {{MIN_TIMESTEP}};
    double (*sv)[35] = new double[NUM_PLANETS][35] {{0.0}};
    double (*r_new)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*v_new)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*a_new)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*j_new)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*s_new)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};
    double (*c_new)[3] = new double[NUM_PLANETS][3] {{0.0,0.0,0.0}};

    // Set the number of omp threads
#ifdef OPENMP
    int max_threads = omp_get_max_threads();
    if(NUM_PLANETS < max_threads)
    {
        omp_set_num_threads(NUM_PLANETS);
    }
    else
    {
        omp_set_num_threads(max_threads);
    }
#endif
    
#pragma omp parallel
{
#ifdef OPENMP
    int id = omp_get_thread_num();
    int nt = omp_get_num_threads();
#else
    int id = 0;
    int nt = 1;
#endif

    // Loop over time
    temporal_loop(r,v,a,j,s,c,r_new,v_new,a_new,j_new,s_new,c_new,ai,ji,si,ci,m,times,timesteps,sv,id,nt);
 
} //omp parallel

    // Cleanup
    delete [] r;
    delete [] v;
    delete [] m;
    delete [] a;
    delete [] j;
    delete [] s;
    delete [] c;
    delete [] ai;
    delete [] ji;
    delete [] si;
    delete [] ci;
    delete [] times;
    delete [] timesteps;
    delete [] sv;
    delete [] r_new;
    delete [] v_new;
    delete [] a_new;
    delete [] j_new;
    delete [] s_new;
    delete [] c_new;
    
    return 0;
}
