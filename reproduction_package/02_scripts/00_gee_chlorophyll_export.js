/**********************************************************************
 * GEE: global coastal-nation chlorophyll-a monthly panel export (for Marine Regions EEZ V12)
 * dependent variable -> SDG 14.1.1 coastal eutrophication | project: ocean-current transboundary SDG spillovers
 **********************************************************************/

// ====== EDIT HERE: paste the Asset ID you get after uploading the EEZ layer ======
var EEZ_ASSET = 'PASTE_YOUR_ASSET_ID_HERE';   // e.g. 'projects/ee-lixxx/assets/eez_v12'

// ====== EEZ V12 fields (usually no change needed) ======
var ISO_FIELD  = 'ISO_TER1';     // territory ISO3
var NAME_FIELD = 'TERRITORY1';   // territory name
var START = '2002-07-01';
var END   = '2021-12-31';
var SCALE = 4616;

// ---- data ----
var eez = ee.FeatureCollection(EEZ_ASSET);
var chl = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI')
            .select('chlor_a').filterDate(START, END);

// ---- build monthly mean images ----
var months = ee.List.sequence(0, ee.Date(END).difference(ee.Date(START),'month').round().subtract(1));
var monthlyChl = ee.ImageCollection.fromImages(months.map(function(m){
  var s = ee.Date(START).advance(m,'month'), e = s.advance(1,'month');
  return chl.filterDate(s,e).mean()
            .set('year', s.get('year')).set('month', s.get('month'))
            .set('system:time_start', s.millis()).rename('chlor_a');
}));

// ---- mean per EEZ x month ----
var results = monthlyChl.map(function(img){
  var stats = img.reduceRegions({
    collection: eez,
    reducer: ee.Reducer.mean().combine(ee.Reducer.count(),'',true),
    scale: SCALE });
  var y = img.get('year'), mo = img.get('month');
  return stats.map(function(f){
    return ee.Feature(null,{
      iso3: f.get(ISO_FIELD), country: f.get(NAME_FIELD),
      year: y, month: mo,
      chlor_a_mean: f.get('mean'), pixel_count: f.get('count') });
  });
}).flatten().filter(ee.Filter.notNull(['chlor_a_mean']));

// ---- export CSV to Google Drive ----
Export.table.toDrive({
  collection: results,
  description: 'coastal_chlorophyll_monthly_by_EEZ',
  fileFormat: 'CSV',
  selectors: ['iso3','country','year','month','chlor_a_mean','pixel_count'] });

// ---- diagnostic prints ----
print('n_months:', months.size());
print('n_EEZ_features:', eez.size());
print('sample EEZ properties (verify field names):', eez.first());
Map.setCenter(-80,22,4);
Map.addLayer(chl.filterDate('2020-01-01','2020-02-01').mean(),
  {min:0,max:5,palette:['blue','green','yellow','red']},'Chl-a 2020-01');
