cd data/01_raw
for ID in 1 2 3 4 5 6 7 8 9 10; do
    echo "######################"
    echo "Downloading subject ${ID}"
    echo "######################"
    wget http://ninapro.hevs.ch/system/files/DB6_Preproc/DB6_s${ID}_a.zip
    wget http://ninapro.hevs.ch/system/files/DB6_Preproc/DB6_s${ID}_b.zip
    unzip DB6_s${ID}_a.zip
    unzip DB6_s${ID}_b.zip
    rm DB6_s${ID}_a.zip
    rm DB6_s${ID}_b.zip
done