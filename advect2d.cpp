
#include <mpi.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <assert.h>

// define new type: matrix of doubles
typedef std::vector<std::vector<double> > dmatrix ;

void init(int N, double dx, dmatrix& C);
void init_mpi(int N, double dx, dmatrix& C, int nprocs, int* coords);
void printMat(int N, dmatrix& C);
void copyAtoB(dmatrix& A, dmatrix& B);
void update(dmatrix& C, dmatrix& C_old, int N, double h, double u, double v);
void printToFile(int N, int step, dmatrix& A);
void printToFile_mpi(dmatrix& C, int N, int subgridlen, int ngrid, int myrank);

int main(int argc, char *argv[]){
    int rflag;
    rflag = 0;
    // parameters: read in
    double NT, T, L, u, v, dx, dt, h;
    int N;
    if (rflag == 1){
        if (argc != 8) {
            printf("Should have 8 args\n");
            return 0;
        }
        N = std::stoi(argv[1]);
        NT = std::stof(argv[2]);
        L = std::stof(argv[3]);
        T = std::stof(argv[4]);
        u = std::stof(argv[5]);
        v = std::stof(argv[6]);
    } else {
        N = 8;
        NT = 20000.0;
        L = 1.0;
       T = 1.0e6;
        u = 5.0e-7;
        v = 2.85e-7;
    }
    printf("N: %d \nNT: %f \nL: %f \nT: %f \nu: %f \nv: %f \n", N, NT, L, T, u, v);
    dx = L/N;
    dt = T/NT;
    h = dx/sqrt(2*(u*u+v*v));
    if (dt>h){
        printf("UNSTABLE! h: %f \ndt: %f\n",h,dt);
    }
    h = dt/(2*dx);

    // SETUP MPI STUFF
    int nprocs, stat, mype, ngrid, subgridlen;
    MPI_Init(&argc, &argv);
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert(stat == MPI_SUCCESS);
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    assert(stat == MPI_SUCCESS);
    assert(fmod(N, sqrt(nprocs)) == 0);

    // setup cartesian communicator 
    MPI_Comm cartcomm;
    int ndim = 2;
    int dims[2] = {0,0};
    int periods[2] = {1,1};
    MPI_Dims_create(nprocs, ndim, dims);
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, 0, &cartcomm);
   
    // assuming (as told we could) that we have a square num of procs 
    // so our virtual layout is ngrid x ngrid procs, each proc has
    // a subgridlen x subgridlen array portion of whole matrix C
    ngrid = sqrt(nprocs);
    subgridlen = N/ngrid;

    // get coords of the subgrid on each proc
    int coords[ndim] ;
    stat = MPI_Cart_coords(cartcomm, mype, ndim, coords);
    
    // initialize matrix portion on each proc
    dmatrix my_C;
    dmatrix my_C_old;
    init_mpi(N, dx, my_C, ngrid, coords);
    init_mpi(N, dx, my_C_old, ngrid, coords);

    printToFile_mpi(my_C, N, subgridlen, ngrid, mype);

    // get initial matrices
   
   /* 
    NT = 10;
    // run for NT timesteps
    for (int step=0;step<NT;step++){
        update(C, C_old, N, h, u, v);
        copyAtoB(C,C_old);
    //    if (step%1000 == 0){
    //        printToFile(N, step, C);
    //    }
    }
    */
    MPI_Finalize();
    return 0;
}

void update(dmatrix& C, dmatrix& C_old, int N, double h, double u, double v){
    int l,r,a,b;
    double temp;
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            // this is all to ensure periodic boundary conditions
            r = i+1;
            l = i-1;
            a = j+1;
            b = j-1;
            if (i==0){
                l = N-1;
            } else if (i==N-1){
                r = 0;
            }
            if (j==0){
                b = N-1;
            } else if (j==N-1){
                a = 0;
            }
            temp = 0.25*(C_old[l][j] + C_old[r][j] + C_old[i][b] + C_old[i][a]);
            C[i][j] = temp - h*(u*(C_old[r][j]-C_old[l][j]) + v*(C_old[i][a] - C_old[i][b]));
        }
    }
    return;
}

// initialize matrix 
void init(int N, double dx, dmatrix& C){
    C.resize(N, std::vector<double>(N));
    double x0, y0, sigx2, sigy2;
    double x,y;
    x0 = dx*N/2;
    y0 = x0;
    sigx2 = 0.25*0.25;
    sigy2 = 0.25*0.25;
    for (int i=0; i<N; i++){
        x = (dx*(i+0.5))-x0;
        for (int j=0; j<N; j++){
            y = (dx*(j+0.5))-y0;
            C[i][j] = exp(-(x*x/(2*sigx2) + y*y/(2*sigy2)));
        }
    }
}

void init_mpi(int N, double dx, dmatrix& C, int subgridlen, int* coords){
    int xa,ya;
    double x0, y0, sigx2, sigy2;
    double x,y;
    xa = subgridlen*coords[0];
    ya = subgridlen*coords[1];
    printf("x,y: (%d,%d)\n", xa, ya);
    C.resize(N, std::vector<double>(N));
    x0 = dx*N/2;
    y0 = x0;
    sigx2 = 0.25*0.25;
    sigy2 = 0.25*0.25;
    for (int i=0; i<N; i++){
        x = (dx*(xa+i+0.5))-x0;
        for (int j=0; j<N; j++){
            y = (dx*(ya+j+0.5))-y0;
            C[i][j] = exp(-(x*x/(2*sigx2) + y*y/(2*sigy2)));
        }
    }
}

// copy values from matrix A to B
void copyAtoB(dmatrix& A, dmatrix& B){
    int nx = A.size();
    for (int i=0;i<nx;i++){
        for (int j=0;j<nx;j++){
            B[i][j] = A[i][j];
        }
    }
    return;
}

// print out matrix values
void printMat(int N, dmatrix& C){
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            printf("%f ",C[i][j]);
        }
        printf("\n");
    }
}

void printToFile(int N, int step, dmatrix& A){
    FILE* file;
    char outname [40];
    int n = sprintf(outname, "out-advec-%d.dat", step);
    file = fopen(outname, "w");
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            fprintf(file, "%f ", A[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
    return;
}

void printToFile_mpi(dmatrix& C, int N, int subgridlen, int ngrid, int myrank){
    if (myrank == 0){
      //  printf("N: %d, ngrid: %d,  subgridlen: %d\n",N,ngrid,subgridlen);
      //  for (int row=0; row<N; row++){
      //      printf("row%d",row);
      //      int p = (row/subgridlen)*ngrid;
      //      for (int pinrow=p; pinrow<(p+ngrid); pinrow++){
      //          for (int k=0; k<subgridlen; k++){
      //              printf(" p%d", pinrow);
      //          }
      //      }
      //      printf("\n");
      //  }

        MPI_Status status;
        double recvbuf[subgridlen];
        FILE* file;
        char outname [40];
        int n = sprintf(outname, "out-advec-init.dat");
        file = fopen(outname, "w");

        int startp, endp, stat;
        // loop over all N rows
        // startp is the first proc containing data in that row
        for (int row=0; row<N; row++){
            startp = (row/subgridlen)*ngrid;
            endp = startp + ngrid;

            printf("row%d  start: %d, end: %d\n",row,startp,endp);
      
            // if startp = 0, print row before receiving from other procs
            // add 1 to startp then? maybe change to send to self also?
            if (startp==myrank){
                for (int j=0; j<subgridlen; j++){
                    fprintf(file, " %f ", C[row][j]);
                }
                startp += 1;
            }

            // now loop over processors from startp to startp+ngrid to get partial row info
            for (int i=startp; i<endp; i++){
                stat = MPI_Recv(&recvbuf, subgridlen, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                assert(stat == MPI_SUCCESS);
                for (int j=0; j<subgridlen; j++){
                    fprintf(file, " %f ", recvbuf[j]);
                }
            }
            fprintf(file, "\n");
        }
        fclose(file);
    } else {
        // have each other proc send its array one row at a time
        for (int row=0; row<subgridlen; row++){
            MPI_Send(&C[row][0], subgridlen, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
        }
    }
    return;
}

