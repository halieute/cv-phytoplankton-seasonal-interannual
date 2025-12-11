
;===============================================================
; NCL script: make_cv_study_area.ncl
; Purpose : Recreate Cabo Verde study-area map with two insets,
;           using shapefiles for Boxes, Islands, and Ocean layers.
; Output  : PNG file "fig_study_area_cv.png"
; Author  : M.N. Souley Souleymane (2025)
;===============================================================

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/gsn_add_shapefile_polylines.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/gsn_add_shapefile_polygons.ncl"

begin
  ;---------------------------
  ; 0) OUTPUT WORKSTATION
  ;---------------------------
  wks = gsn_open_wks("png","fig_study_area_cv")  ; output: fig_study_area_cv.png

  ;---------------------------
  ; 1) COLOR DEFINITIONS
  ;---------------------------
  ; Ocean (light blue), Islands (orange), Boxes (blue)
  oceanRGB  = (/189./255., 220./255., 244./255./)
  islandRGB = (/228./255., 111./255.,  44./255./)
  boxRGB    = (/ 54./255.,  83./255., 175./255./)

  ;---------------------------
  ; 2) MAP EXTENTS
  ;---------------------------
  ; Main Cabo Verde extent (adjust if needed)
  minLat = 14.0
  maxLat = 19.5
  minLon = -27.0
  maxLon = -22.0

  ; Overview #1: West Africa
  ov1_minLat =  0.0
  ov1_maxLat = 30.0
  ov1_minLon = -20.0
  ov1_maxLon =  10.0

  ; Overview #2: Wider Cabo Verde box
  ov2_minLat = 10.0
  ov2_maxLat = 25.0
  ov2_minLon = -33.0
  ov2_maxLon = -15.0

  ;---------------------------
  ; 3) SHAPEFILE PATHS
  ;---------------------------
  ; Replace with your actual file paths (ESRI Shapefile format)
  shp_islands = "data/islands.shp"   ; polygons of all CV islands
  shp_ocean   = "data/ocean.shp"     ; a single polygon for ocean background (optional)
  shp_boxes   = "data/boxes.shp"     ; polygons of CV1..CV5 boxes (or polylines)

  ;---------------------------
  ; 4) LABELS FOR CV1..CV5
  ;---------------------------
  ; Provide label positions (lon/lat) near the boxes or their centroids.
  cv_names = (/"CV1","CV2","CV3","CV4","CV5"/)
  cv_lon   = (/ -26.50, -24.00, -23.00, -25.20, -22.80 /)   ; <-- update
  cv_lat   = (/  16.80,  16.20,  15.20,  17.10,  16.50 /)   ; <-- update

  ;---------------------------
  ; 5) PAGE LAYOUT (VIEWPORTS)
  ;---------------------------
  ; Right-side: main map; left-side: two insets stacked
  m_vpx = 0.35   ; X of main viewport (page fraction)
  m_vpy = 0.88   ; Y of main viewport
  m_vpw = 0.60   ; width
  m_vph = 0.82   ; height

  i1_vpx = 0.10  ; inset 1 (upper-left)
  i1_vpy = 0.88
  i1_vpw = 0.23
  i1_vph = 0.38

  i2_vpx = 0.10  ; inset 2 (lower-left)
  i2_vpy = 0.45
  i2_vpw = 0.23
  i2_vph = 0.38

  ;---------------------------
  ; 6) COMMON MAP RESOURCES
  ;---------------------------
  common = True
  common@gsnDraw               = False
  common@gsnFrame              = False
  common@mpProjection          = "Mercator"
  common@mpFillOn              = True
  common@mpOutlineOn           = True
  common@mpPerimOn             = True
  common@mpGridAndLimbOn       = False
  common@mpOutlineBoundarySets = "National"    ; insets: show country borders
  common@mpGeophysicalLineThicknessF = 1.3
  common@mpNationalLineThicknessF    = 1.2
  common@mpOceanFillColor      = oceanRGB      ; ocean color
  common@mpLandFillColor       = "White"
  common@mpInlandWaterFillColor= "White"

  ; Slightly larger tick fonts for the main map
  common@tmXBFormat                = "f"
  common@tmYLFormat                = "f"
  common@tmXBLabelFontHeightF      = 0.012
  common@tmYLLabelFontHeightF      = 0.012
  common@tmXBLabelFont             = "helvetica"
  common@tmYLLabelFont             = "helvetica"

  ;---------------------------
  ; 7) INSET #1: WEST AFRICA
  ;---------------------------
  i1 = True
  i1 = common
  i1@vpXF      = i1_vpx
  i1@vpYF      = i1_vpy
  i1@vpWidthF  = i1_vpw
  i1@vpHeightF = i1_vph
  i1@mpMinLatF = ov1_minLat
  i1@mpMaxLatF = ov1_maxLat
  i1@mpMinLonF = ov1_minLon
  i1@mpMaxLonF = ov1_maxLon
  i1@tmXBLabelFontHeightF = 0.010
  i1@tmYLLabelFontHeightF = 0.010

  map_i1 = gsn_csm_map(wks,i1)

  ; draw a rectangle to indicate inset #2 coverage
  lnres = True
  lnres@gsLineThicknessF = 2.0
  lnres@gsLineColor      = "Black"
  xbox = (/ ov2_minLon, ov2_maxLon, ov2_maxLon, ov2_minLon, ov2_minLon /)
  ybox = (/ ov2_minLat, ov2_minLat, ov2_maxLat, ov2_maxLat, ov2_minLat /)
  id_i1_box = gsn_add_polyline(wks,map_i1,xbox,ybox,lnres)

  ;---------------------------
  ; 8) INSET #2: WIDER CV VIEW
  ;---------------------------
  i2 = True
  i2 = common
  i2@vpXF      = i2_vpx
  i2@vpYF      = i2_vpy
  i2@vpWidthF  = i2_vpw
  i2@vpHeightF = i2_vph
  i2@mpMinLatF = ov2_minLat
  i2@mpMaxLatF = ov2_maxLat
  i2@mpMinLonF = ov2_minLon
  i2@mpMaxLonF = ov2_maxLon
  i2@tmXBLabelFontHeightF = 0.010
  i2@tmYLLabelFontHeightF = 0.010

  map_i2 = gsn_csm_map(wks,i2)

  ; draw a rectangle to indicate main-map coverage
  xbox2 = (/ minLon, maxLon, maxLon, minLon, minLon /)
  ybox2 = (/ minLat, minLat, maxLat, maxLat, minLat /)
  id_i2_box = gsn_add_polyline(wks,map_i2,xbox2,ybox2,lnres)

  ;---------------------------
  ; 9) MAIN MAP: CABO VERDE
  ;---------------------------
  m = True
  m = common
  m@vpXF      = m_vpx
  m@vpYF      = m_vpy
  m@vpWidthF  = m_vpw
  m@vpHeightF = m_vph
  m@mpMinLatF = minLat
  m@mpMaxLatF = maxLat
  m@mpMinLonF = minLon
  m@mpMaxLonF = maxLon

  ; emphasize axis tick labels on main map
  m@tmXBLabelFontHeightF = 0.0125
  m@tmYLLabelFontHeightF = 0.0125

  map_main = gsn_csm_map(wks,m)

  ;---------------------------
  ; 9a) OCEAN BACKGROUND (optional shapefile)
  ;---------------------------
  if (isfilepresent(shp_ocean)) then
    oceanres = True
    oceanres@gsFilled    = True
    oceanres@gsFillColor = oceanRGB
    oceanres@gsEdgesOn   = False
    id_ocean = gsn_add_shapefile_polygons(wks,map_main,shp_ocean,oceanres)
  end if

  ;---------------------------
  ; 9b) ISLANDS (polygons)
  ;---------------------------
  if (isfilepresent(shp_islands)) then
    polyres = True
    polyres@gsFilled         = True
    polyres@gsFillColor      = islandRGB
    polyres@gsEdgesOn        = True
    polyres@gsEdgeColor      = "Black"
    polyres@gsEdgeThicknessF = 1.2
    id_islands = gsn_add_shapefile_polygons(wks,map_main,shp_islands,polyres)
  else
    print("WARNING: islands shapefile not found: " + shp_islands)
  end if

  ;---------------------------
  ; 9c) BOXES (polygons or polylines)
  ;---------------------------
  if (isfilepresent(shp_boxes)) then
    boxres = True
    boxres@gsFilled         = False         ; outline only (like QGIS diamonds)
    boxres@gsEdgesOn        = True
    boxres@gsEdgeColor      = boxRGB
    boxres@gsEdgeThicknessF = 2.0
    ; If your boxes.shp are polygons and you want filled diamonds:
    ; boxres@gsFilled    = True
    ; boxres@gsFillColor = boxRGB
    id_boxes = gsn_add_shapefile_polylines(wks,map_main,shp_boxes,boxres)
    ; NOTE: use gsn_add_shapefile_polygons if your file is polygon geometry and you want filled shapes.
  else
    print("WARNING: boxes shapefile not found: " + shp_boxes)
  end if

  ;---------------------------
  ; 9d) LABELS FOR CV1..CV5
  ;---------------------------
  txres = True
  txres@txFontHeightF = 0.014
  txres@txFont        = "helvetica"
  txres@txJust        = "CenterLeft"

  do i=0,dimsizes(cv_lon)-1
    ; Offset slightly to avoid overlapping outlines
    gsn_add_text(wks,map_main,cv_names(i),cv_lon(i)+0.08,cv_lat(i)+0.06,txres)
  end do

  ;---------------------------
  ; 9e) SCALE BAR (~200 km)
  ;---------------------------
  ; Approximate km per degree of longitude at mid-lat (Mercator)
  midlat   = 17.0
  km_per_deg_lon = 111.32 * cos(midlat*3.14159/180.0)
  span_km  = 200.0
  span_deg = span_km / km_per_deg_lon

  sb_lon0 = maxLon - 0.30 - span_deg
  sb_lon1 = maxLon - 0.30
  sb_lat  = minLat + 0.35

  sbres = True
  sbres@gsLineThicknessF = 4.0
  sbres@gsLineColor      = "Black"
  gsn_add_polyline(wks,map_main,(/sb_lon0,sb_lon1/),(/sb_lat,sb_lat/),sbres)

  sbtx = True
  sbtx@txFontHeightF = 0.012
  sbtx@txFont        = "helvetica"
  gsn_add_text(wks,map_main,"0",            sb_lon0,                 sb_lat-0.15, sbtx)
  gsn_add_text(wks,map_main,"200 km",       (sb_lon0+sb_lon1)/2.,    sb_lat-0.15, sbtx)

  ;---------------------------
  ; 9f) SIMPLE LEGEND (Boxes, Islands, Ocean)
  ;---------------------------
  lg_lon = minLon + 0.45
  lg_lat = maxLat - 0.30

  ; Boxes symbol: draw a diamond marker
  mkres = True
  mkres@gsMarkerIndex = 12      ; diamond
  mkres@gsMarkerSizeF = 0.020
  mkres@gsMarkerColor = boxRGB
  gsn_add_polymarker(wks,map_main,lg_lon,lg_lat,mkres)
  lgtx = True
  lgtx@txFontHeightF = 0.014
  lgtx@txFont        = "helvetica"
  gsn_add_text(wks,map_main,"Boxes", lg_lon+0.25, lg_lat, lgtx)

  ; Islands sample patch
  px = (/ lg_lon, lg_lon+0.15, lg_lon+0.15, lg_lon, lg_lon /)
  py = (/ lg_lat-0.25, lg_lat-0.25, lg_lat-0.45, lg_lat-0.45, lg_lat-0.25 /)
  pg = True
  pg@gsFilled    = True
  pg@gsFillColor = islandRGB
  gsn_add_polygon(wks,map_main,px,py,pg)
  gsn_add_text(wks,map_main,"Islands", lg_lon+0.25, lg_lat-0.35, lgtx)

  ; Ocean sample patch
  px2 = px
  py2 = py - 0.35
  pg2 = True
  pg2@gsFilled    = True
  pg2@gsFillColor = oceanRGB
  gsn_add_polygon(wks,map_main,px2,py2,pg2)
  gsn_add_text(wks,map_main,"Ocean", lg_lon+0.25, lg_lat-0.70, lgtx)

  ;---------------------------
  ; 9g) ATTRIBUTION TEXT
  ;---------------------------
  atx = True
  atx@txFontHeightF = 0.012
  atx@txFont        = "helvetica"
  atx@txJust        = "BottomCenter"
  gsn_add_text(wks,map_main,"Data source: OCHA / OSM / Natural Earth  |  Design: M.N. Souleymane 2025", \
               (minLon+maxLon)/2.0, minLat+0.12, atx)

  ;---------------------------
  ; 10) DRAW & FRAME
  ;---------------------------
  draw(map_i1)
  draw(map_i2)
  draw(map_main)
  frame(wks)

end