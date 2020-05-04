@echo off
git fetch

FOR /F %%i IN ('git branch -a') DO (
    echo %count% %%i | findstr ^remotes
)
set /p rama=escribe la rama a actualizar no escriba remotes/origin/ solamente lo que sigue a continuacion:
git checkout --force %rama%
git pull
set /p salir=pulse enter para salir