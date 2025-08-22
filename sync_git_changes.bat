@echo off
echo ========================================
echo      Syncing Git Changes
echo ========================================
echo.

cd /d "%~dp0"

echo Current branch:
git branch

echo.
echo Changes to be committed:
git status

echo.
echo Do you want to commit all changes to the current branch? (Y/N)
set /p confirm=

if /i "%confirm%"=="Y" (
    echo.
    echo Adding all changes...
    git add .
    
    echo.
    echo Committing changes...
    git commit -m "Update voice agent with fixes and optimizations"
    
    echo.
    echo Changes committed successfully!
    
    echo.
    echo Do you want to push these changes to GitHub? (Y/N)
    set /p push=
    
    if /i "%push%"=="Y" (
        echo.
        echo Pushing changes to GitHub...
        git push
        echo.
        echo Changes pushed successfully!
    )
) else (
    echo.
    echo Operation cancelled. No changes were committed.
)

echo.
echo ========================================
pause