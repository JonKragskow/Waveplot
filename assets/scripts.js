
function downloadURI(uri, name) {
    var link = document.createElement("a");
    link.download = name;
    link.href = uri;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function updateViewText(prefix, viewer) {
    document.getElementById(prefix.concat("_view_x")).value = viewer.getView()[0];
    document.getElementById(prefix.concat("_view_y")).value = viewer.getView()[1];
    document.getElementById(prefix.concat("_view_z")).value = viewer.getView()[2];
    document.getElementById(prefix.concat("_view_zoom")).value = viewer.getView()[3];
    document.getElementById(prefix.concat("_view_qx")).value = viewer.getView()[4];
    document.getElementById(prefix.concat("_view_qy")).value = viewer.getView()[5];
    document.getElementById(prefix.concat("_view_qz")).value = viewer.getView()[6];
    document.getElementById(prefix.concat("_view_qw")).value = viewer.getView()[7];
}

