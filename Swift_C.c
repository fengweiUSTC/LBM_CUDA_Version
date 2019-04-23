#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> 

#define Imax 512
#define Jmax 512

#pragma warning(disable:4996)

int  Ox = 100, Oy = 100, R = 40;

float  tau = 1.0;
float  rho1 = 1.90, rho2 = 5.20;
float  T = 0.560, kappa = 0.010, a = 9.00/49.00, b = 2.00/21.00;
float  w[9] = { 4.0/9.0, \
				1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, \
				1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0 };
float  f[9][Imax][Jmax], f_temp[9][Imax][Jmax], f_eq[9][Imax][Jmax];
float  u_node[Imax][Jmax], v_node[Imax][Jmax], vel_node[Imax][Jmax], 
		P[Imax][Jmax], rho_node[Imax][Jmax];
float  error;
int     obst[Imax][Jmax];

void initialize_func(void);
void getf_eq(void);
void stream_func(void);
void collision_func(void);
void conver_func(void);
void output(int counter);

int main()
{
	clock_t start, finish;
	int inter_num = 40000, counter;
	start = clock();

	//初始化
	initialize_func();
	output(0);

	//迭代计算
	for (counter = 1; counter <= inter_num; ++counter)
	{
		printf("%d\n", counter);
		//对流步
		stream_func();

		//收敛判据
		//conver_func();

		/*if ((int)fmod(counter, 1000) == 0)
		{
			conver_func();
			printf("Counter=%d, Error=%f\n", counter, error);
		}*/

		//碰撞步
		collision_func();

		/*if ((int)fmod(counter, 1000) == 0)  
			output(counter);

		if (counter==100)  output(counter);*/
	}
	
	finish = clock();
	output(counter);
	printf("\nSpent %.3f seconds in calculating.\n", (float)(finish - start) / CLOCKS_PER_SEC);

	system("pause");

	return 0;
}

void initialize_func(void)
{
	int  i, j;
	float disturb;

	//f_temp[9][Imax][Jmax] = { 0.0 };
	srand(1234);

	
	for (j = 0; j < Jmax; ++j)
	{
		for (i = 0; i < Imax; ++i)
		{
			/*
			if ((i - Ox)*(i - Ox) + (j - Oy)*(j - Oy) <= R*R)
			{
				rho_node[i][j] = rho1;
			}
			else
			{
				rho_node[i][j] = rho2;
			}
			*/

			//srand(1234);
			disturb = (float)rand() / (float)(RAND_MAX);
			rho_node[j][i] = 3.0 + disturb;
			//rho_node[j][i] = j * Imax + i;
			//printf("disturb=%f, rho_node=%f\n", disturb, rho_node[i][j]);
		}
	}

	getf_eq();

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			f[0][i][j] = f_eq[0][i][j];
			f[1][i][j] = f_eq[1][i][j];
			f[2][i][j] = f_eq[2][i][j];
			f[3][i][j] = f_eq[3][i][j];
			f[4][i][j] = f_eq[4][i][j];
			f[5][i][j] = f_eq[5][i][j];
			f[6][i][j] = f_eq[6][i][j];
			f[7][i][j] = f_eq[7][i][j];
			f[8][i][j] = f_eq[8][i][j];
		}
	}
}

void getf_eq(void)
{
	int  i, j, i_w, i_e, j_n, j_s;
	float  rho_xgrad, rho_ygrad, rho_laplace;
	float  A0, A1, A2, B1, B2, C0, C1, C2, D1, D2, \
			Gxx1, Gxx2, Gyy1, Gyy2, Gxy1, Gxy2;
	float  vel_dire[8], Cs_squ = 1.0 / 3.0;
	float  u, v, vel_node;

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			P[i][j] = rho_node[i][j] * T / (1.0 - rho_node[i][j] * b)\
						- a*pow(rho_node[i][j], 2);

			if (i != 0)
				i_w = i - 1;
			else
				i_w = Imax - 1;

			if (i != Imax - 1)
				i_e = i + 1;
			else
				i_e = 0;

			if (j != 0)
				j_s = j - 1;
			else
				j_s = Jmax - 1;

			if (j != Jmax - 1)
				j_n = j + 1;
			else
				j_n = 0;
			

				//rho_xgrad = 1.0 / Cs_squ*(w(1)*rho_node(i_e, j) - w(3)*rho_node(i_w, j) + w(5)*rho_node(i_e, j_n) + w(8)*rho_node(i_e, j_s) - w(6)*rho_node(i_w, j_n) - w(7)*rho_node(i_w, j_s))//
			rho_xgrad = (rho_node[i_e][j] - rho_node[i_w][j]) / 2.00;
				//rho_ygrad = 1.0 / Cs_squ*(w(2)*rho_node(i, j_n) - w(4)*rho_node(i, j_s) + w(5)*rho_node(i_e, j_n) - w(8)*rho_node(i_e, j_s) + w(6)*rho_node(i_w, j_n) - w(7)*rho_node(i_w, j_s))//
			rho_ygrad = (rho_node[i][j_n] - rho_node[i][j_s]) / 2.00;
			rho_laplace = (rho_node[i_e][j] + rho_node[i][j_n] + rho_node[i_w][j] + rho_node[i][j_s])*4.0 / 6.0 +\
						  (rho_node[i_e][j_n] + rho_node[i_w][j_n] + rho_node[i_w][j_s] +rho_node[i_e][j_s]) / 6.00 -\
						  20.0 / 6.0*rho_node[i][j];

			A1 = 1.0 / 3.0*(P[i][j] - kappa*rho_node[i][j] * rho_laplace);
			A2 = A1 / 4.0;
			A0 = rho_node[i][j] - 5.0 / 3.0*(P[i][j] - kappa*rho_node[i][j] * rho_laplace);
			B2 = rho_node[i][j] / 12.0;
			B1 = 4.0*B2;
			C2 = -rho_node[i][j] / 24.0;
			C1 = 4.0*C2;
			C0 = -2.0*rho_node[i][j] / 3.0;
			D1 = rho_node[i][j] / 2.0;
			D2 = rho_node[i][j] / 8.0;

			Gxx1 = kappa / 4.0*(pow(rho_xgrad,2) - pow(rho_ygrad,2));
			Gxx2 = Gxx1 / 4.0;
			Gyy1 = -Gxx1;
			Gyy2 = -Gxx2;
			Gxy1 = kappa / 2.0*rho_xgrad*rho_ygrad;
			Gxy2 = Gxy1 / 4.0;

			u = ((f_temp[1][i][j] - f_temp[3][i][j]) + (f_temp[5][i][j] - f_temp[7][i][j]) + (f_temp[8][i][j] - f_temp[6][i][j])) / rho_node[i][j];
			v = ((f_temp[2][i][j] - f_temp[4][i][j]) + (f_temp[5][i][j] - f_temp[7][i][j]) + (f_temp[6][i][j] - f_temp[8][i][j])) / rho_node[i][j];
			vel_node = u * u + v * v;

			if (obst[i][j] == 0)
			{
				vel_dire[0] =  u;
				vel_dire[1] =  v;
				vel_dire[2] = -u;
				vel_dire[3] = -v;
				vel_dire[4] =  u + v;
				vel_dire[5] = -u + v;
				vel_dire[6] = -u - v;
				vel_dire[7] =  u - v;

				f_eq[0][i][j]  = A0 + C0*vel_node;
				f_eq[1][i][j] = A1 + B1*vel_dire[0] + C1*vel_node +\
								D1*vel_dire[0] * vel_dire[0] + Gxx1;
				f_eq[2][i][j] = A1 + B1*vel_dire[1] + C1*vel_node +\
								D1*vel_dire[1] * vel_dire[1] + Gyy1;
				f_eq[3][i][j] = A1 + B1*vel_dire[2] + C1*vel_node +\
								D1*vel_dire[2] * vel_dire[2] + Gxx1;
				f_eq[4][i][j] = A1 + B1*vel_dire[3] + C1*vel_node +\
								D1*vel_dire[3] * vel_dire[3] + Gyy1;
				f_eq[5][i][j] = A2 + B2*vel_dire[4] + C2*vel_node +\
								D2*vel_dire[4] * vel_dire[4] + Gxx2 + Gyy2 + 2.0*Gxy2;
				f_eq[6][i][j] = A2 + B2*vel_dire[5] + C2*vel_node +\
								D2*vel_dire[5] * vel_dire[5] + Gxx2 + Gyy2 - 2.0*Gxy2;
				f_eq[7][i][j] = A2 + B2*vel_dire[6] + C2*vel_node +\
								D2*vel_dire[6] * vel_dire[6] + Gxx2 + Gyy2 + 2.0*Gxy2;
				f_eq[8][i][j] = A2 + B2*vel_dire[7] + C2*vel_node +\
								D2*vel_dire[7] * vel_dire[7] + Gxx2 + Gyy2 - 2.0*Gxy2;
			}
		}
	}
}

void stream_func(void)
{
	int i, j, i_w, i_e, j_n, j_s;

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			if (i != 0)
				i_w = i - 1;
			else
				i_w = Imax - 1;

			if (i != Imax - 1)
				i_e = i + 1;
			else
				i_e = 0;

			if (j != 0)
				j_s = j - 1;
			else
				j_s = Jmax - 1;

			if (j != Jmax - 1)
				j_n = j + 1;
			else
				j_n = 0;

			f_temp[0][i  ][j  ] = f[0][i][j];
			f_temp[1][i_e][j  ] = f[1][i][j];
			f_temp[2][i  ][j_n] = f[2][i][j];
			f_temp[3][i_w][j  ] = f[3][i][j];
			f_temp[4][i  ][j_s] = f[4][i][j];
			f_temp[5][i_e][j_n] = f[5][i][j];
			f_temp[6][i_w][j_n] = f[6][i][j];
			f_temp[7][i_w][j_s] = f[7][i][j];
			f_temp[8][i_e][j_s] = f[8][i][j];
		}
	}
}

void collision_func(void)
{
	int  i, j, alpha;

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			rho_node[i][j] = 0.0;
			for (alpha = 0; alpha <= 8; ++alpha)
			{
				rho_node[i][j] = rho_node[i][j] + f_temp[alpha][i][j];
			}
		}
	}

	getf_eq();

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			for (alpha = 0; alpha <= 8; ++alpha)
			{
				f[alpha][i][j] = f_temp[alpha][i][j] - 1.0 / tau*(f_temp[alpha][i][j] - f_eq[alpha][i][j]);
			}
		}
	}
}

void conver_func(void)
{
	int i, j, alpha;
	float p_next, rho_node;

	error = 0.0;

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			if (obst[i][j] == 0)
			{
				rho_node = 0.0;
				for (alpha = 0; alpha <= 8; ++alpha)
				{
					rho_node = rho_node + f_temp[alpha][i][j];
				}
				p_next = rho_node*T / (1.0 - rho_node*b) - a*pow(rho_node, 2);
       			error = error + fabs(p_next - P[i][j]) / p_next;
			}
		}
	}
}

void output(int counter)
{
	char name1[50], name2[10];
	char tec_title[50], field[100];
	char radiu[10];

	int  i, j;

	FILE *fp1;

	sprintf_s(name1, 50, "result\\%-6dtwophase_swift.plt", counter);
	_gcvt_s(radiu, 5, T, 2);
	
	//strcpy_s(name1, 20, "twophase_swift.plt");
	fopen_s(&fp1, name1, "w+");
	fputs("VARIABLES=\"X\",\"Y\",\"density\"\n", fp1);
	sprintf_s(tec_title, 50, "ZONE I=%d,J=%d,F=POINT\n", Imax, Jmax);
	fputs(tec_title, fp1);
	//write(10, "(A7,I3,A3,I3,A8)") "ZONE I=", Imax, ",J=", Jmax, ",F=POINT";

	for (i = 0; i < Imax; ++i)
	{
		for (j = 0; j < Jmax; ++j)
		{
			sprintf_s(field, 100, "%4d %4d %17.9f\n", i, Jmax-1-j, rho_node[j][i]);
			fputs(field, fp1);
		}
	}
	fclose(fp1);

	/*name2 = "check_isotropy"//radiu//".dat"
	open(11, file = trim(name2))
	do j = 1, Jmax
	j = (Jmax / 2)
	do i = 1, Imax
	write(11, "(I3,E17.9)") i, rho_node(i, j)
	end do
	end do
	close(11)*/
}