    module global
    implicit none

    integer :: Ox=101,Oy=101,R=40
    integer,parameter :: Imax=201,Jmax=201

    real(kind=8) :: tau=1.d0
    real(kind=8) :: rho1=1.9d0,rho2=5.2d0
    real(kind=8) :: T=0.54d0,kappa=0.01d0,a=9.0d0/49.0d0,b=2.0d0/21.0d0
    real(kind=8) :: w(0:8)=(/ 4.d0/9.d0,1.d0/9.d0,1.d0/9.d0,1.d0/9.d0,1.d0/9.d0,1.d0/36.d0,1.d0/36.d0,1.d0/36.d0,1.d0/36.d0 /)
    real(kind=8),save :: f(0:8,Imax,Jmax),f_temp(0:8,Imax,Jmax),f_eq(0:8,Imax,Jmax)
    real(kind=8),save :: u_node(Imax,Jmax),v_node(Imax,Jmax),vel_node(Imax,Jmax),P(Imax,Jmax),rho_node(Imax,Jmax)
    real(kind=8)   error
    logical obst(Imax,Jmax)

    contains

    subroutine initialize_func()
    implicit none

    integer i,j
    real disturb

    f_temp(:,:,:)=0.0d0

    do j=1,Jmax
        do i=1,Imax

            if ( (i-Ox)*(i-Ox)+(j-Oy)*(j-Oy) <= R*R) then
            !if ( i<=(Imax/2) ) then
                rho_node(i,j)=rho1
            else
                rho_node(i,j)=rho2
            end if

            
            !call RANDOM_SEED()
            !call RANDOM_NUMBER(disturb)
            !rho_node(i,j)=3.d0+0.01d0*disturb
            

        end do
    end do

    call getf_eq()

    do j=1,Jmax
        do i=1,Imax
            f(0,i,j)=f_eq(0,i,j)
            f(1,i,j)=f_eq(1,i,j)
            f(2,i,j)=f_eq(2,i,j)
            f(3,i,j)=f_eq(3,i,j)
            f(4,i,j)=f_eq(4,i,j)
            f(5,i,j)=f_eq(5,i,j)
            f(6,i,j)=f_eq(6,i,j)
            f(7,i,j)=f_eq(7,i,j)
            f(8,i,j)=f_eq(8,i,j)
        end do
    end do

    return
    end subroutine initialize_func

    subroutine getf_eq()
    implicit none

    integer i,j,alpha,i_w,i_e,j_n,j_s
    real(kind=8) :: rho_xgrad,rho_ygrad,rho_laplace
    real(kind=8) :: A0,A1,A2,B1,B2,C0,C1,C2,D1,D2,Gxx1,Gxx2,Gyy1,Gyy2,Gxy1,Gxy2
    real(kind=8) :: vel_dire(8),Cs_squ=1.d0/3.d0

    do j=1,Jmax
        do i=1,Imax

            P(i,j)=rho_node(i,j)*T/(1-rho_node(i,j)*b) - a*rho_node(i,j)**2

            i_w=Imax-mod(Imax+1-i,Imax)
            i_e=mod(i,Imax)+1
            j_n=mod(j,Jmax)+1
            j_s=Jmax-mod(Jmax+1-j,Jmax)

            !rho_xgrad = 1.d0/Cs_squ*( w(1)*rho_node(i_e,j) - w(3)*rho_node(i_w,j) + w(5)*rho_node(i_e,j_n) + w(8)*rho_node(i_e,j_s) - w(6)*rho_node(i_w,j_n) - w(7)*rho_node(i_w,j_s)  )
            rho_xgrad = ( rho_node(i_e,j)-rho_node(i_w,j) )/2.0d0
            !rho_ygrad = 1.d0/Cs_squ*( w(2)*rho_node(i,j_n) - w(4)*rho_node(i,j_s) + w(5)*rho_node(i_e,j_n) - w(8)*rho_node(i_e,j_s) + w(6)*rho_node(i_w,j_n) - w(7)*rho_node(i_w,j_s)  )
            rho_ygrad = ( rho_node(i,j_n)-rho_node(i,j_s) )/2.0d0
            rho_laplace = (rho_node(i_e,j) + rho_node(i,j_n) + rho_node(i_w,j) + rho_node(i,j_s))*4.d0/6.d0 + (rho_node(i_e,j_n) + rho_node(i_w,j_n) + rho_node(i_w,j_s) + rho_node(i_e,j_s))/6.0d0-20.d0/6.d0*rho_node(i,j)

            A1=1.d0/3.d0*(P(i,j)-kappa*rho_node(i,j)*rho_laplace)
            A2=A1/4.0d0
            A0=rho_node(i,j)-5.d0/3.d0*(P(i,j)-kappa*rho_node(i,j)*rho_laplace)
            B2=rho_node(i,j)/12.d0
            B1=4.d0*B2
            C2=-rho_node(i,j)/24.d0
            C1=4.d0*C2
            C0=-2.d0*rho_node(i,j)/3.d0
            D1=rho_node(i,j)/2.d0
            D2=rho_node(i,j)/8.d0

            Gxx1=kappa/4.d0*(rho_xgrad**2-rho_ygrad**2)
            Gxx2=Gxx1/4.d0
            Gyy1=-Gxx1
            Gyy2=-Gxx2
            Gxy1=kappa/2.d0*rho_xgrad*rho_ygrad
            Gxy2=Gxy1/4.d0

            u_node(i,j)=( (f_temp(1,i,j)-f_temp(3,i,j)) + (f_temp(5,i,j)-f_temp(7,i,j)) + (f_temp(8,i,j)-f_temp(6,i,j)))/rho_node(i,j)
            v_node(i,j)=( (f_temp(2,i,j)-f_temp(4,i,j)) + (f_temp(5,i,j)-f_temp(7,i,j)) + (f_temp(6,i,j)-f_temp(8,i,j)))/rho_node(i,j)
            vel_node(i,j)= u_node(i,j)*u_node(i,j)+v_node(i,j)*v_node(i,j)

            if (.not. obst(i,j)) then
                vel_dire(1)= u_node(i,j)
                vel_dire(2)=               v_node(i,j)
                vel_dire(3)=-u_node(i,j)
                vel_dire(4)=             - v_node(i,j)
                vel_dire(5)= u_node(i,j) + v_node(i,j)
                vel_dire(6)=-u_node(i,j) + v_node(i,j)
                vel_dire(7)=-u_node(i,j) - v_node(i,j)
                vel_dire(8)= u_node(i,j) - v_node(i,j)

                f_eq(0,i,j)= A0 + C0*vel_node(i,j)
                f_eq(1,i,j)= A1 + B1*vel_dire(1) + C1*vel_node(i,j) + D1*vel_dire(1)*vel_dire(1) + Gxx1
                f_eq(2,i,j)= A1 + B1*vel_dire(2) + C1*vel_node(i,j) + D1*vel_dire(2)*vel_dire(2) + Gyy1
                f_eq(3,i,j)= A1 + B1*vel_dire(3) + C1*vel_node(i,j) + D1*vel_dire(3)*vel_dire(3) + Gxx1
                f_eq(4,i,j)= A1 + B1*vel_dire(4) + C1*vel_node(i,j) + D1*vel_dire(4)*vel_dire(4) + Gyy1
                f_eq(5,i,j)= A2 + B2*vel_dire(5) + C2*vel_node(i,j) + D2*vel_dire(5)*vel_dire(5) + Gxx2 + Gyy2 + 2.d0*Gxy2
                f_eq(6,i,j)= A2 + B2*vel_dire(6) + C2*vel_node(i,j) + D2*vel_dire(6)*vel_dire(6) + Gxx2 + Gyy2 - 2.d0*Gxy2
                f_eq(7,i,j)= A2 + B2*vel_dire(7) + C2*vel_node(i,j) + D2*vel_dire(7)*vel_dire(7) + Gxx2 + Gyy2 + 2.d0*Gxy2
                f_eq(8,i,j)= A2 + B2*vel_dire(8) + C2*vel_node(i,j) + D2*vel_dire(8)*vel_dire(8) + Gxx2 + Gyy2 - 2.d0*Gxy2
            end if
        end do
    end do
    return
    end subroutine getf_eq

    subroutine stream_func()
    implicit none

    integer i,j,i_w,i_e,j_n,j_s

    do j=1,Jmax
        do i=1,Imax
            i_w=Imax-mod(Imax+1-i,Imax)
            i_e=mod(i,Imax)+1
            j_n=mod(j,Jmax)+1
            j_s=Jmax-mod(Jmax+1-j,Jmax)

            f_temp(0, i  , j  )=f(0,i,j)
            f_temp(1, i_e, j  )=f(1,i,j)
            f_temp(2, i  , j_n)=f(2,i,j)
            f_temp(3, i_w, j  )=f(3,i,j)
            f_temp(4, i  , j_s)=f(4,i,j)
            f_temp(5, i_e, j_n)=f(5,i,j)
            f_temp(6, i_w, j_n)=f(6,i,j)
            f_temp(7, i_w, j_s)=f(7,i,j)
            f_temp(8, i_e, j_s)=f(8,i,j)

        end do
    end do

    return
    end subroutine stream_func

    subroutine collision_func()
    implicit none

    integer i,j,alpha

    do j=1,Jmax
        do i=1,Imax
            rho_node(i,j)=0.d0
            do alpha=0,8
                rho_node(i,j)=rho_node(i,j)+f_temp(alpha,i,j)
            end do
        end do
    end do

    call getf_eq()

    do j=1,Jmax
        do i=1,Imax
            do alpha=0,8
                f(alpha,i,j)= f_temp(alpha,i,j)-1.d0/tau*( f_temp(alpha,i,j)-f_eq(alpha,i,j))
            end do
        end do
    end do

    return
    end subroutine collision_func

    subroutine conver_func()
    implicit none

    integer i,j,alpha
    real(kind=8) :: p_next,rho_node

    error=0.d0
    do j=1,Jmax
        do i=1,Imax
            if (.not. obst(i,j)) then
                rho_node=0.d0

                do alpha=0,8
                    rho_node=rho_node+f_temp(alpha,i,j)
                end do

                p_next=rho_node*T/(1-rho_node*b) - a*rho_node**2

                error=error+abs(p_next-p(i,j))/p_next
            end if
        end do
    end do
    return
    end subroutine conver_func

    subroutine output(counter)
    implicit none

    integer counter
    character(len=6) num
    character(len=50) name1,name2
    character(len=4) radiu
    integer i,j

    write(num,"(I6.6)") counter
    write(radiu,"(F4.2)") T
    name1="new_twophase_swift"//num//".plt"
    open(10,file=trim(name1))
    write(10,*) "VARIABLES=""X"",""Y"",""P"",""density"""
    write(10,"(A7,I3,A3,I3,A8)") "ZONE I=",Imax,",J=",Jmax,",F=POINT"

    do j=1,Jmax
        do i=1,Imax
            write(10,"(I4,I4,E17.9,E17.9)") i,j,P(i,j),rho_node(i,j)
        end do
    end do
    close(10)

    !name2="check_isotropy"//radiu//".dat"
    !open(11,file=trim(name2))
    !do j=1,Jmax
    !j=(Jmax/2)
    !    do i=1,Imax
    !        write(11,"(I3,E17.9)") i,rho_node(i,j)
    !    end do
    !end do
    !close(11)
    return
    end subroutine output

    end module

    program main
    use global
    implicit none

    integer :: inter_num=30000,counter

    ! 定义边界和障碍物
    !call obstacle_func()

    ! 初始化
    call initialize_func()
    call output(0)

    ! 迭代计算
    do counter=1,inter_num

        ! 对流步
        call stream_func()

        ! 边界节点处的对流步
        !call boundary_func()

        ! 收敛判据
        call conver_func()
        !if (error<1d-6) exit
        !if (mod(counter,1000) == 0)  
        write(*,*) counter,error

        ! 碰撞步
        call collision_func()

        ! 输出全部节点处的速度等信息
        if (mod(counter,500) == 0) call output(counter)
    end do
    
    pause

    end program main