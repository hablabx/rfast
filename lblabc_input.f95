! to compile:
!   python -m numpy.f2py -c lblabc_input.f95 -m lblabc_input
!
! then in python:
!   import lblabc_input
!
SUBROUTINE read_lblabc(iuabc,npg,ntg,nwn,p,tatm,wn,abc)
!
   IMPLICIT NONE
!
!   INTEGER, PARAMETER :: iuabc = 101
   INTEGER, PARAMETER :: mxp   = 20
   INTEGER, PARAMETER :: mxt   = 20
   INTEGER, PARAMETER :: mxwn  = 100000
!
   INTEGER :: i,j,k,l,igtyp,nbgas,stat
   INTEGER, INTENT(INOUT) :: npg,ntg,nwn
   INTEGER, INTENT(IN) :: iuabc
   INTEGER, DIMENSION(50) :: ibgas
   REAL(8), DIMENSION(mxwn), INTENT(INOUT) :: wn
   REAL(8), DIMENSION(mxp,mxt,mxwn), INTENT(INOUT) :: abc
   REAL, DIMENSION(mxp,mxt) :: abc0
   REAL :: wn0,dwn,mxwn0
   REAL(8), DIMENSION(mxp), INTENT(INOUT) :: p
   REAL, DIMENSION(mxp) :: p0
   REAL(8), DIMENSION(mxp,mxt), INTENT(INOUT) :: tatm
   REAL, DIMENSION(mxp,mxt) :: tatm0
   REAL, DIMENSION(mxp,mxt) :: rg
!   CHARACTER(LEN=120), INTENT(IN) :: fnIN
!
!  open for reading
!
!   OPEN(iuabc,file=fnIN,form='unformatted',status='old',action='read')
!
!  read header information
!
   READ(iuabc,IOSTAT=stat) npg,ntg,igtyp,nbgas,(ibgas(i),i=1,nbgas),wn0,mxwn0
   READ(iuabc,IOSTAT=stat) (p0(k),k=1,npg)
   READ(iuabc,IOSTAT=stat) ((rg(k,i),i=1,nbgas),k=1,npg)
   READ(iuabc,IOSTAT=stat) ((tatm0(k,l),k=1,npg),l=1,ntg)
!
!  save for returning to python
!
   DO k=1,npg
     p(k) = p0(k)
     DO l=1,ntg
       tatm(k,l) = tatm0(k,l)
     END DO
   END DO
!
!  read opacities
!
   j = 1
   DO
     READ(iuabc,IOSTAT=stat) wn0,dwn,((abc0(k,l),k=1,npg),l=1,ntg)
     wn(j) = wn0
     DO k=1,npg
       DO l=1,ntg
         abc(k,l,j) = abc0(k,l)
       END DO
     END DO
     j = j + 1  
     IF (stat /= 0) EXIT
   END DO
!
   nwn = j-1
!
!  close file
!
   CLOSE(iuabc)
!
END SUBROUTINE
!
!  subroutine for opening .abs output file
!
SUBROUTINE open_lblabc(iuabc,fnIN)
   IMPLICIT NONE
   INTEGER, INTENT(IN) :: iuabc
   CHARACTER(LEN=300), INTENT(IN) :: fnIN
!
   OPEN(iuabc,file=fnIN,form='unformatted',status='old',action='read')
END SUBROUTINE
