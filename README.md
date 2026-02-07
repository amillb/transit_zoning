# Mapping High-Quality Transit

This code identifies high-quality transit areas (HQTAs) as defined under California state law. In general, HQTAs are within one half mile of a “major transit stop”: a rail, ferry, or bus rapid transit station, or the intersection of frequent bus routes.

The definitions under state law leave much ambiguity. For example, what constitutes an "intersection" of bus routes? Over what period should frequency be calculated? Is the half mile measured from the platform or from the edge of the parcel on which the station sits (including parking lots)? We discuss these ambiguities in a UCLA Institute of Transportation Studies [interactive storymap](http://its.ucla.edu/major-transit-stops).

The code ingests GTFS files (in `.zip` format) and outputs polygons of HQTAs under both a restrictive ("minimal") and expansive ("maximal") interpretation of these ambiguities. It also ingests supplementary files that give the locations of Amtrak stations, subway station entrances, and more.

## Requirements
The code has been tested with the following packages. Other versions may also work. 
```
python == 3.12.9
numpy == 2.2.4
pandas == 2.2.3
geopandas == 1.0.1
partridge == 1.1.2
```
To install with Anaconda:

`conda install numpy=2.2.4 pandas=2.2.3 geopandas=1.0.1 pip`

`pip install partridge==1.1.2`

## How to run
You'll need to obtain and download GTFS data for each agency for the relevant years. (We include a few example GTFS files here, but license restrictions mean that we can't redistribute the GTFS data here on GitHub.) You can delete the directories for years that you don't need to analyze.

The analysis also uses data on BRT routes, Amtrak stations, subway station entrances, station parcels, and planned transit. These are all included in this repository in the `other_transit` directory. See `sources.txt` for sources and more details of the purpose of each file.

Organize the input data in the same way as this repository:
- transit_data/
  - gtfs/
    - 2014/
      - agency1_feed.zip
      - agency2_feed.zip
    - 2020/
      - agency1_feed.zip
      - agency2_feed.zip
    - 2025/
      - agency1_feed.zip
      - agency2_feed.zip
  - other_transit/
      - files are included in this repository
     
You can edit the code to add or remove the files that the code expects in `other_transit`. For example, you might want to include planned transit from other MPOs.

To run from the command line:
`python transit_zoning.py`

Or from within Python:
`run transit_zoning.py`

## Credits
Conceptualization and analysis: Jacob Wasserman, Aaron Barrall, Adam Millard-Ball, and Amy Lee

Code: Adam Millard-Ball and Daniel Sjoholm, with assistance from Marcel Moran

## More information
Questions? Need help? Interested in a custom analysis?

Contact [Adam Millard-Ball](https://millardball.its.ucla.edu)

## Citation
If you use this code, please cite:
Wasserman, J., Barrall, A., Millard-Ball, A., and Lee, A. (2026). “Stop” and Think about It: How the Different Interpretations of What Counts as a “Major Transit Stop” in California Make a Difference. http://its.ucla.edu/major-transit-stops
