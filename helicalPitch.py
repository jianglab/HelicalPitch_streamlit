#!/usr/bin/env python

# $Id$

import numpy as np
import pandas as pd
import streamlit as st

def main():
    title = "HelicalPitch: determine helical pitch/twist using 2D Classification info"
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)

    with st.expander(label="README", expanded=False):
            st.write("This is a Web App that helps the user determine helical pitch/twist using 2D classification info. When two segments on the same filament are assigned to the same 2D class, it means that two segments have same rotation angle around the helical axis and the distance between this pair of segments will be equal to pitch/csym. Here, we find all those pairs, collect the pair distances, and then plot the histogram of the pair distances. If the 2D classification quality is good, the histogram should show prominent peaks of equal spacing with the spacing = pitch/csym.  \n  \n*NOTE: the uploaded files are **strictly confidential**. The developers of this app do not have access to the files*")

    col1, [col2, col3] = st.sidebar, st.columns([0.5,1])

    with col1:
        params = None
        input_modes_params = {0:"upload", 1:"url"} 
        help_params = "Only RELION star and cryoSPARC cs formats are supported"
        input_mode_params = st.radio(label="How to obtain the Class2D parameter file:", options=list(input_modes_params.keys()), format_func=lambda i:input_modes_params[i], index=1, horizontal=True, help=help_params, key="input_mode_params")
        if input_mode_params == 0: # "upload a star or cs file":
            label = "Upload the class2d parameters in a RELION star or cryoSPARC cs file"
            help = None
            fileobj = st.file_uploader(label, type=['star', "cs"], help=help, key="upload_params")
            if fileobj is None:
                return
            if fileobj.name.endswith(".cs"):    # cryoSPARC
                label = "Upload the cryoSPARC pass through cs file"
                fileobj_cs_pass_through = st.file_uploader(label, type=["cs"], help=help, key="upload_params_cs_pass_through")
                if fileobj_cs_pass_through is None:
                    return
                params = get_class2d_params_from_uploaded_file(fileobj, fileobj_cs_pass_through)
            else: # RELION
                params = get_class2d_params_from_uploaded_file(fileobj)
        else:   # download from a URL
            params = None
            label = "Download URL for a RELION star or cryoSPARC cs file"
            help = None
            url_default = "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/768px/run_it020_data.star"
            url = st.text_input(label, value=url_default, help=help, key="url_params")
            if url is None:
                return
            if url.endswith(".cs"):
                label = "Download URL for the cryoSPARC pass through cs file"
                url_cs_pass_through = st.text_input(label, value=None, help=help, key="url_params_cs_pass_through")
                if url_cs_pass_through is None:
                    return
                params = get_class2d_params_from_url(url, url_cs_pass_through)
            else:
                params = get_class2d_params_from_url(url)
        if params is None:
            return

        nHelices, nClasses = get_number_helices_classes(params)
        st.write(f"*{len(params):,} particles | {nHelices:,} filaments | {nClasses} classes*")

        st.divider()

        input_modes_classes = {0:"upload", 1:"url"} 
        help_classes = "Only MRC (*\*.mrcs*) format is supported"
        input_mode_classes = st.radio(label="How to obtain the class average images:", options=list(input_modes_classes.keys()), format_func=lambda i:input_modes_classes[i], index=1, horizontal=True, help=help_classes, key="input_mode_classes")
        if input_mode_classes == 0: # "upload a MRC file":
            label = "Upload the class averages in MRC format (.mrcs, .mrc)"
            help = None
            fileobj = st.file_uploader(label, type=['mrcs', 'mrc'], help=help, key="upload_classes")
            if fileobj is not None:
                data_all, apix = get_class2d_from_uploaded_file(fileobj)
            else:
                return
        else:   # download from a URL
            label = "Download URL for a RELION or cryoSPARC Class2D output mrc(s) file"
            help = None
            url_default = "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/768px/run_it020_classes.mrcs"
            url = st.text_input(label, value=url_default, help=help, key="url_classes")
            if url is not None:
                data_all, apix = get_class2d_from_url(url)
            else:
                return

        n, ny, nx = data_all.shape
        st.write(f"*{n} classes: {nx}x{ny} pixels | {round(apix,4)} Å/pixel*")

        abundance = get_class_abundance(params, n)
        display_seq = np.arange(n, dtype=int)
        
        ignore_blank = st.checkbox("Ignore blank classes", value=True, key="ignore_blank")
        sort_abundance = st.checkbox("Sort the classes by abundance", value=True, key="sort_abundance")
        if sort_abundance:
            display_seq = np.argsort(abundance)[::-1]

        with st.expander(label="Choose a class", expanded=True):
            from st_clickable_images import clickable_images
            images = []
            images_displayed = []
            for i in display_seq:
                if abundance[i] or not ignore_blank:
                    images.append(encode_numpy(data_all[i], vflip=True))
                    images_displayed.append(i)
            thumbnail_size = 128
            n_per_row = 400//thumbnail_size
            with st.container(height=min(500, len(images)*thumbnail_size//n_per_row), border=False):
                image_index = clickable_images(
                    images,
                    titles=[f"Class {i+1}: {abundance[i]:,} particles" for i in display_seq if abundance[i] or not ignore_blank],
                    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                    img_style={"margin": "1px", "height": f"{thumbnail_size}px"},
                    key=f"class_display"
                )
                image_index = max(0, image_index)
        class_index = images_displayed[image_index]

        apix_particle = st.number_input("Pixel size of particles", min_value=0.1, max_value=100.0, value=apix, format="%.3f", help="Å/pixel", key="apix_particle")
        # ["micrograph_blob/psize_A", "rlnMicrographPixelSize", "rlnMicrographOriginalPixelSize", "blob/psize_A", "rlnImagePixelSize"]
        apix_micrograph_auto = get_pixel_size(params, return_source=True)
        if apix_micrograph_auto is None:
            apix_micrograph_auto = apix_particle
            apix_micrograph_auto_source = "particle pixel size"
        else:
            apix_micrograph_auto, apix_micrograph_auto_source = apix_micrograph_auto
        if apix_micrograph_auto_source in ["micrograph_blob/psize_A", "rlnMicrographPixelSize"]:
            value = apix_micrograph_auto
            placeholder = None
        else:
            value = None
            placeholder = f"{apix_micrograph_auto:.3f}"
        apix_micrograph = st.number_input("Pixel size of micrographs for particle extraction", min_value=0.1, max_value=100.0, value=value, placeholder=placeholder, format="%.3f", help=f"Å/pixel. Make sure a correct value is specified", key="apix_micrograph")
        if apix_micrograph is None:
            msg = "Please manually input the pixel value for the micrographs used for particle picking/extraction. "
            if apix_micrograph_auto_source in ["rlnMicrographOriginalPixelSize"]:
                msg += f"The placeholder value {apix_micrograph_auto:.3f} was from the **{apix_micrograph_auto_source}** parameter in the input star file. Note that the **{apix_micrograph_auto_source}** value is for the original movies, NOT for the movie averages that were used to extract the particles. If you have binned the movies during motion correction, you should multply {apix_micrograph_auto:.3f} by the bin factor"
                st.warning(msg)
            elif apix_micrograph_auto_source in ["blob/psize_A", "rlnImagePixelSize", "particle pixel size"]:
                msg += f"The placeholder value {apix_micrograph_auto:.3f} was based on **{apix_micrograph_auto_source}**. Note that If you have downscaled the particles during particle extraction, you should divide {apix_micrograph_auto:.3f} by the downscaling factor used during particle extraction"
                st.warning(msg)

        if apix_micrograph is None:
            st.stop()

    with col2:        
        params = update_particle_locations(params, apix_micrograph)
        with st.spinner(f"Selecting filaments in Class {class_index+1}"):
            helices = select_class(params, class_index, apix_micrograph)

        data = data_all[class_index]

        image_label = f"Class {class_index+1}: {abundance[class_index]:,} segments | {len(helices):,} filaments"
        fig = create_image_figure(data, apix, apix, title=image_label, title_location="above", plot_width=None, plot_height=None, x_axis_label=None, y_axis_label=None, tooltips=None, show_axis=False, show_toolbar=False, crosshair_color="white")
        st.bokeh_chart(fig, use_container_width=True)
        
        with st.spinner("Checking filament length"):
            filement_lengths = get_filament_length(helices, apix_particle*nx)

        log_y = True
        title = f"Filament Lengths: Class {class_index+1}"
        xlabel = "Filament Legnth (Å)"
        ylabel = "# of Filaments"
        fig = plot_histogram(filement_lengths, title=title, xlabel=xlabel, ylabel=ylabel, bins=50, log_y=log_y)
        st.bokeh_chart(fig, use_container_width=True)
        
        min_len = st.number_input("Select filaments of a minimal length (Å)", min_value=0.0, max_value=np.max(filement_lengths), value=0.0, format="%.0f", help="Only use filaments of at least this length for subsequent pair-distance calculation", key="min_len")
        
        helices_retained, n_ptcls = select_helices_by_length(helices, filement_lengths, min_len)
        st.write(f"*{len(helices_retained)} filaments ≥ {min_len} Å in length → {n_ptcls} segments*")
        
        rise = st.number_input("Helical rise (Å)", min_value=0.01, max_value=apix_particle*nx, value=4.75, format="%.2f", help="helical rise", key="rise")

    with col3:
        with st.spinner("Computing the histogram"):
            pair_dists = process_one_class(helices_retained)
        title = f"Pair Distances: Class {class_index+1}"
        xlabel = "Pair Distance (Å)"
        ylabel = "# of Pairs"
        fig = plot_histogram(pair_dists, title=title, xlabel=xlabel, ylabel=ylabel, bins=100, log_y=log_y, show_pitch_twist=dict(rise=rise, csyms=(1,2,3)), multi_crosshair=True)
        st.bokeh_chart(fig, use_container_width=True)
        st.write("**How to interpretate the histogram:** an informative histogram should have clear peaks with equal spacing. If so, hover your mouse pointer on the first prominent peak to the right side of the primary peak at the origin to show the twist values assuming the pair-distance is the helical pitch (adjusted for the cyclic symmetries around the helical axis). If the histogram does not show clear peaks, it indicates that the Class2D quality is bad. You might consider redoing the Class2D task with longer extracted segments (>0.5x helical pitch) from longer filaments (> 1x pitch)")

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu/HelicalPitch). Report problems to [HelicalPitch@GitHub](https://github.com/jianglab/HelicalPitch/issues)*")

    return

#@st.cache_data(show_spinner=False)
def process_one_class(helices):
    dists_same_class = []
    for _, segments in helices:
        loc_x = segments['rlnCoordinateX'].values.astype(float)
        loc_y = segments['rlnCoordinateY'].values.astype(float)
        psi = segments['rlnAnglePsi'].values.astype(float)

        # Calculate pairwise distances only for segments with the same polarity
        mask = np.abs((psi[:, None] - psi + 180) % 360 - 180) < 90
        dx = loc_x[:, None] - loc_x
        dy = loc_y[:, None] - loc_y
        distances = np.sqrt(dx**2 + dy**2)
        distances = distances[mask]
        dists_same_class.extend(distances[distances > 0])  # Exclude zero distances (self-distances)
    if not dists_same_class:
        return None
    else:
        return np.sort(dists_same_class)

def plot_histogram(data, title, xlabel, ylabel, bins=50, log_y=True, show_pitch_twist={}, multi_crosshair=False):     
    from bokeh.plotting import ColumnDataSource, figure
    hist, edges = np.histogram(data, bins=bins)
    hist_linear = hist
    if log_y:
        hist = np.log10(1+hist)
    
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    fig = figure(title=title, tools=tools, aspect_ratio=1)
    center = (edges[:-1]+edges[1:])/2
    source_data = dict(top=hist, left=edges[:-1], right=edges[1:], center=center, hist_linear=hist_linear)
    if show_pitch_twist:
        rise = show_pitch_twist["rise"]
        csyms = show_pitch_twist["csyms"]
        tmp = {}
        for csym in csyms:
            tmp[f"C{csym}"] = 360/(center*csym/rise)
        source_data.update(tmp)
    source = ColumnDataSource(data=source_data)
        
    hist_plot = fig.quad(top='top', bottom=0, left='left', right='right', fill_color="skyblue", source=source)
    fig.xaxis.axis_label = xlabel
    fig.yaxis.axis_label = f"log10({ylabel})" if log_y else ylabel
    fig.title.align = "center"
    fig.title.text_font_size = "18px"
    fig.title.text_font_style = "normal"
    from bokeh.models.tools import HoverTool
    tooltips = [(xlabel, "@center{0} (@left{0}-@right{0})"), (ylabel, "@hist_linear")]
    if show_pitch_twist:
        for csym in show_pitch_twist["csyms"]:
            tooltips.append((f"Twist for C{csym} (°)", f"@C{csym}{{0.00}}"))
    hist_hover = HoverTool(renderers=[hist_plot], tooltips=tooltips)
    fig.add_tools(hist_hover)
    fig.hover[0].attachment="vertical"

    if multi_crosshair:
        from bokeh.models import Span, CustomJS
        n_crosshairs = 20
        crosshairs = []
        for i in range(2, n_crosshairs):
            crosshair = Span(visible=False, dimension='height', line_dash='dashed')
            fig.add_layout(crosshair)
            crosshairs.append(crosshair)
        callback_code = "\n".join([f"crosshair_{i}.location = x*{i};" for i in range(2, n_crosshairs)])
        callback = CustomJS(args={f"crosshair_{i}": crosshairs[i-2] for i in range(2, n_crosshairs)}, 
            code=f"""
                var x = cb_obj.x;
                {callback_code}
            """)
        fig.js_on_event('mousemove', callback)      
        callback_code = "\n".join([f"crosshair_{i}.visible = false;" for i in range(2, n_crosshairs)])
        callback = CustomJS(args={f"crosshair_{i}": crosshairs[i-2] for i in range(2, n_crosshairs)}, 
            code=f"""
                {callback_code}
            """)
        fig.js_on_event('mouseleave', callback)
        callback_code = "\n".join([f"crosshair_{i}.visible = true;" for i in range(2, n_crosshairs)])
        callback = CustomJS(args={f"crosshair_{i}": crosshairs[i-2] for i in range(2, n_crosshairs)}, 
            code=f"""
                {callback_code}
            """)
        fig.js_on_event('mouseenter', callback)
    return fig

#@st.cache_data(show_spinner=False)
def select_helices_by_length(helices, lengths, min_len):
    helices_retained = []
    n_ptcls = 0
    for gi, (gn, g) in enumerate(helices):
        if lengths[gi] >= min_len:
            n_ptcls += len(g)
            helices_retained.append((gn, g))
    return helices_retained, n_ptcls

#@st.cache_data(show_spinner=False)
def get_filament_length(helices, particle_box_length):
    filement_lengths = []
    for gn, g in helices:
        track_lengths = g["rlnHelicalTrackLengthAngst"].astype(float).values
        length = track_lengths.max() - track_lengths.min() + particle_box_length
        filement_lengths.append(length)
    return filement_lengths

@st.cache_data(show_spinner=False)
def select_class(params, class_index, apix_micrograph):
    particles = params.loc[params["rlnClassNumber"].astype(int)==class_index+1, :]
    helices = list(particles.groupby(["rlnMicrographName", "rlnHelicalTubeID"]))
    return helices

@st.cache_data(show_spinner=False)
def get_class_abundance(params, nClass):
    abundance = np.zeros(nClass, dtype=int)
    for gn, g in params.groupby("rlnClassNumber"):
        abundance[int(gn)-1] = len(g)
    return abundance
    
@st.cache_data(show_spinner=False)
def update_particle_locations(params, apix_micrograph):
    ret = params.copy()
    ret['rlnCoordinateX'] = params.loc[:, 'rlnCoordinateX'].astype(float)*apix_micrograph
    ret['rlnCoordinateY'] = params.loc[:, 'rlnCoordinateY'].astype(float)*apix_micrograph
    if 'rlnOriginXAngst' in params and 'rlnOriginYAngst' in params:
        ret.loc[:, 'rlnCoordinateX'] -= params.loc[:, 'rlnOriginXAngst'].astype(float)
        ret.loc[:, 'rlnCoordinateY'] -= params.loc[:, 'rlnOriginYAngst'].astype(float)
    return ret

@st.cache_data(show_spinner=False)
def get_number_helices_classes(params):
    nHelices = len(list(params.groupby(["rlnMicrographName", "rlnHelicalTubeID"])))
    nClasses = len(params["rlnClassNumber"].unique())
    return  nHelices, nClasses 
    
def create_image_figure(image, dx, dy, title="", title_location="above", plot_width=None, plot_height=None, x_axis_label='x', y_axis_label='y', tooltips=None, show_axis=True, show_toolbar=True, crosshair_color="white", aspect_ratio=None):
    from bokeh.plotting import ColumnDataSource, figure
    h, w = image.shape
    if aspect_ratio is None:
        if plot_width and plot_height:
            aspect_ratio = plot_width/plot_height
        else:
            aspect_ratio = w*dx/(h*dy)
    tools = 'box_zoom,crosshair,pan,reset,save,wheel_zoom'
    fig = figure(title_location=title_location, 
        frame_width=plot_width, frame_height=plot_height, 
        x_axis_label=x_axis_label, y_axis_label=y_axis_label,
        x_range=(-w//2*dx, (w//2-1)*dx), y_range=(-h//2*dy, (h//2-1)*dy), 
        tools=tools, aspect_ratio=aspect_ratio)
    fig.grid.visible = False
    if title:
        fig.title.text=title
        fig.title.align = "center"
        fig.title.text_font_size = "18px"
        fig.title.text_font_style = "normal"
    if not show_axis: fig.axis.visible = False
    if not show_toolbar: fig.toolbar_location = None

    source_data = ColumnDataSource(data=dict(image=[image], x=[-w//2*dx], y=[-h//2*dy], dw=[w*dx], dh=[h*dy]))
    from bokeh.models import LinearColorMapper
    color_mapper = LinearColorMapper(palette='Greys256')    # Greys256, Viridis256
    image = fig.image(source=source_data, image='image', color_mapper=color_mapper,
                x='x', y='y', dw='dw', dh='dh'
            )

    from bokeh.models.tools import HoverTool, CrosshairTool
    if not tooltips:
        tooltips = [("x", "$xÅ"), ('y', '$yÅ'), ('val', '@image')]
    image_hover = HoverTool(renderers=[image], tooltips=tooltips)
    fig.add_tools(image_hover)
    fig.hover[0].attachment="vertical"
    crosshair = [t for t in fig.tools if isinstance(t, CrosshairTool)]
    if crosshair: 
        for ch in crosshair: ch.line_color = crosshair_color
    return fig

def get_pixel_size(data, attrs=["micrograph_blob/psize_A", "rlnMicrographPixelSize", "rlnMicrographOriginalPixelSize", "blob/psize_A", "rlnImagePixelSize"], return_source=False):
    try:
        sources = [ data.attrs["optics"] ]
    except:
        sources = []
    sources += [data]
    for source in sources:
        for attr in attrs:
            if attr in source:
                if attr in ["rlnImageName", "rlnMicrographName"]:
                    import mrcfile, pathlib
                    folder = pathlib.Path(data["starFile"].iloc[0])
                    if folder.is_symlink(): folder = folder.readlink()
                    folder = folder.resolve().parent 
                    filename = source[attr].iloc[0].split("@")[-1]
                    filename = str((folder / "../.." / filename).resolve())
                    with mrcfile.open(filename, header_only=True) as mrc:
                        apix = float(mrc.voxel_size.x)
                else:
                    apix = float(source[attr].iloc[0])
                if return_source: return apix, attr
                else: return apix
    return None

@st.cache_data(show_spinner=False)
def assign_segment_id(data, inter_segment_distance):
    assert "rlnHelicalTrackLengthAngst" in data
    tmp = data.loc[:, "rlnHelicalTrackLengthAngst"].astype(float) / inter_segment_distance
    err = (tmp - tmp.round()).abs()
    if np.sum(np.where(err > 0.1))>0:
        print(f"WARNING: it appears that the helical segments were extracted with different inter-segment distances")
    helical_segment_id = tmp.round().astype(int)
    return helical_segment_id

@st.cache_data(show_spinner=False)
def estimate_inter_segment_distance(data):
    # data must have been sorted by micrograph, rlnHelicalTubeID, and rlnHelicalTrackLengthAngst
    helices = data.groupby(['rlnMicrographName', "rlnHelicalTubeID"], sort=False)

    import numpy as np
    dists_all = []
    for _, particles in helices:
        if len(particles)<2: continue
        dists = np.sort(particles["rlnHelicalTrackLengthAngst"].astype(float).values)
        dists = dists[1:] - dists[:-1]
        dists_all.append(dists)
    dists_all = np.hstack(dists_all)
    dist_seg = np.median(dists_all)   # Angstrom
    return dist_seg

def encode_numpy(img, hflip=False, vflip=False):
    if img.dtype != np.dtype('uint8'):
        vmin, vmax = img.min(), img.max()
        tmp = (255*(img-vmin)/(vmax-vmin)).astype(np.uint8)
    else:
        tmp = img
    if hflip:
        tmp = tmp[:, ::-1]
    if vflip:
        tmp = tmp[::-1, :]
    import io, base64
    from PIL import Image
    pil_img = Image.fromarray(tmp)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64, {encoded}"

@st.cache_data(show_spinner=False)
def get_class2d_from_uploaded_file(fileobj):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        return get_class2d_from_file(temp.name)

@st.cache_data(show_spinner=False)
def get_class2d_from_url(url):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        st.error(f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
        st.stop()
    data = get_class2d_from_file(fileobj.name)
    return data

@st.cache_data(show_spinner=False)
def get_class2d_from_file(classFile):
    import mrcfile
    with mrcfile.open(classFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, apix

@st.cache_data(show_spinner=False)
def get_class2d_params_from_uploaded_file(fileobj, fileobj_cs_pass_through=None):
    import os, tempfile
    orignal_filename = fileobj.name
    suffix = os.path.splitext(orignal_filename)[-1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(fileobj.read())
        if fileobj_cs_pass_through is None:
            return get_class2d_params_from_file(temp.name)
        else:
            orignal_filename_cs_pass_through = fileobj_cs_pass_through.name
            suffix = os.path.splitext(orignal_filename_cs_pass_through)[-1]
            with tempfile.NamedTemporaryFile(suffix=suffix) as temp_cs_pass_through:
                temp_cs_pass_through.write(fileobj_cs_pass_through.read())
                return get_class2d_params_from_file(temp.name, temp_cs_pass_through.name)

@st.cache_data(show_spinner=False)
def get_class2d_params_from_url(url, url_cs_pass_through=None):
    url_final = get_direct_url(url)    # convert cloud drive indirect url to direct url
    fileobj = download_file_from_url(url_final)
    if fileobj is None:
        st.error(f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
        st.stop()
    if url_cs_pass_through is None:
        data = get_class2d_params_from_file(fileobj.name)
        return data
    url_final_cs_pass_through = get_direct_url(url_cs_pass_through)    # convert cloud drive indirect url to direct url
    fileobj_cs_pass_through = download_file_from_url(url_final_cs_pass_through)
    if fileobj_cs_pass_through is None:
        st.error(f"ERROR: {url_cs_pass_through} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview")
        st.stop()
    data = get_class2d_params_from_file(fileobj.name, fileobj_cs_pass_through.name)
    return data    

@st.cache_data(show_spinner=False)
def get_class2d_params_from_file(params_file, cryosparc_pass_through_file=None):
    with st.spinner("Reading parameters"):
        if params_file.endswith(".star"):
            params = star_to_dataframe(params_file)
        elif params_file.endswith(".cs"):
            assert cryosparc_pass_through_file is not None
            params = cs_to_dataframe(params_file, cryosparc_pass_through_file)
    required_attrs = np.unique("rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnCoordinateX rlnCoordinateY rlnClassNumber rlnAnglePsi".split())
    missing_attrs = [attr for attr in required_attrs if attr not in params]
    if missing_attrs:
        st.error(f"ERROR: parameters {missing_attrs} are not available")
        st.stop()
    distseg = estimate_inter_segment_distance(params)
    params["segmentid"] = assign_segment_id(params, distseg)
    return params

@st.cache_data(show_spinner=False)
def star_to_dataframe(starFile):
    import pandas as pd
    from gemmi import cif
    star = cif.read_file(starFile)
    if len(star) == 2:
        optics = pd.DataFrame()
        for item in star[0]:
            for tag in item.loop.tags:
                value = star[0].find_loop(tag)
                optics[tag.strip('_')] = np.array(value)
    else:
        optics = None

    data = pd.DataFrame()
    for item in star[-1]:
        for tag in item.loop.tags:
            value = star[-1].find_loop(tag)
            data[tag.strip('_')] = np.array(value)
    
    if optics is not None:
        data.attrs["optics"] = optics

    data["starFile"] = starFile
    return data

@st.cache_data(show_spinner=False)
def cs_to_dataframe(cs_file, cs_pass_through_file):
    cs = np.load(cs_file)
    df_cs = pd.DataFrame.from_records(cs.tolist(), columns=cs.dtype.names)
    cs_passthrough = np.load(cs_pass_through_file)
    df_cs_passthrough = pd.DataFrame.from_records(cs_passthrough.tolist(), columns=cs_passthrough.dtype.names)
    data = pd.concat([df_cs, df_cs_passthrough], axis=1)
    data = data.loc[:, ~data.columns.duplicated()]
    # rlnImageName rlnHelicalTubeID rlnHelicalTrackLengthAngst rlnCoordinateX rlnCoordinateY rlnClassNumber rlnAnglePsi
    ret = pd.DataFrame()
    if "blob/idx" in data and "blob/path" in data:
        ret["rlnImageName"] = (data['blob/idx'].astype(int)+1).map('{:06d}'.format) + "@" + data['blob/path'].str.decode('utf-8')
    if "blob/psize_A" in data:
        ret["rlnImagePixelSize"] = data["blob/psize_A"]
        ret["blob/psize_A"] = data["blob/psize_A"]
    if "micrograph_blob/path" in data:
        ret["rlnMicrographName"] = data["micrograph_blob/path"]
    if "micrograph_blob/psize_A" in data:
        ret["rlnMicrographPixelSize"] = data["micrograph_blob/psize_A"]
        ret["micrograph_blob/psize_A"] = data["micrograph_blob/psize_A"]
    if "location/micrograph_path" in data:
        ret["rlnMicrographName"] = data["location/micrograph_path"]
    if "location/center_x_frac" in data and "location/center_y_frac" in data and "location/micrograph_shape" in data:
        locations = pd.DataFrame(data["location/micrograph_shape"].tolist())
        my = locations.iloc[:, 0]
        mx = locations.iloc[:, 1]
        ret["rlnCoordinateX"] = (data["location/center_x_frac"] * mx).astype(float).round(2)
        ret["rlnCoordinateY"] = (data["location/center_y_frac"] * my).astype(float).round(2)
    if "filament/filament_uid" in data:
        if "blob/path" in data:
            if data["filament/filament_uid"].min()>1000:
                micrographs = data.groupby(["blob/path"])
                for _, m in micrographs:
                    mapping = {v : i+1 for i, v in enumerate(sorted(m["filament/filament_uid"].unique()))} 
                    ret.loc[m.index, "rlnHelicalTubeID"] = m["filament/filament_uid"].map(mapping)
            else:
                ret.loc[:, "rlnHelicalTubeID"] = data["filament/filament_uid"].astype(int)

            if "filament/position_A" in data:
                filaments = data.groupby(["blob/path", "filament/filament_uid"])
                for _, f in filaments:
                    val = f["filament/position_A"].astype(np.float32).values
                    val -= np.min(val)
                    ret.loc[f.index, "rlnHelicalTrackLengthAngst"] = val.round(2)
        else:
            mapping = {v : i+1 for i, v in enumerate(sorted(data["filament/filament_uid"].unique()))} 
            ret.loc[:, "rlnHelicalTubeID"] = data["filament/filament_uid"].map(mapping)
    if "filament/filament_pose" in data:
        ret.loc[:, "rlnAnglePsi"] = np.round(-np.rad2deg(data["filament/filament_pose"]), 1)
    # 2D class assignments
    if "alignments2D/class" in data:
        ret["rlnClassNumber"] = data["alignments2D/class"].astype(int) + 1
    if "alignments2D/shift" in data:
        shifts = pd.DataFrame(data["alignments2D/shift"].tolist()).round(2)
        ret["rlnOriginX"] = -shifts.iloc[:, 0]
        ret["rlnOriginY"] = -shifts.iloc[:, 1]
    if "alignments2D/pose" in data:
        ret["rlnAnglePsi"] = -np.rad2deg(data["alignments2D/pose"]).round(2)
    return ret

def download_file_from_url(url):
    import tempfile
    import requests
    try:
        filesize = get_file_size(url)
        local_filename = url.split('/')[-1]
        suffix = '.' + local_filename
        fileobj = tempfile.NamedTemporaryFile(suffix=suffix)
        with st.spinner(f'Downloading {url} ({filesize/2**20:.1f} MB)'):
            with requests.get(url) as r:
                r.raise_for_status()  # Check for request success
                fileobj.write(r.content)
        return fileobj
    except requests.exceptions.RequestException as e:
        return None

@st.cache_data(show_spinner=False)
def get_file_size(url):
    import requests
    response = requests.head(url)
    if 'Content-Length' in response.headers:
        file_size = int(response.headers['Content-Length'])
        return file_size
    else:
        return None
    
def get_direct_url(url):
    import re
    if url.startswith("https://drive.google.com/file/d/"):
        hash = url.split("/")[5]
        return f"https://drive.google.com/uc?export=download&id={hash}"
    elif url.startswith("https://app.box.com/s/"):
        hash = url.split("/")[-1]
        return f"https://app.box.com/shared/static/{hash}"
    elif url.startswith("https://www.dropbox.com"):
        if url.find("dl=1")!=-1: return url
        elif url.find("dl=0")!=-1: return url.replace("dl=0", "dl=1")
        else: return url+"?dl=1"
    elif url.find("sharepoint.com")!=-1 and url.find("guestaccess.aspx")!=-1:
        return url.replace("guestaccess.aspx", "download.aspx")
    elif url.startswith("https://1drv.ms"):
        import base64
        data_bytes64 = base64.b64encode(bytes(url, 'utf-8'))
        data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    else:
        return url

def import_with_auto_install(packages, scope=locals()):
    import importlib, site, subprocess, sys
    if isinstance(packages, str):
        packages = [packages]
    for package in packages:
        if ":" in package:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = importlib.import_module(package_import_name)
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_pip_name])
            importlib.reload(site)
            scope[package_import_name] = importlib.import_module(package_import_name)
import_with_auto_install("numpy pandas PIL:pillow bokeh:bokeh==2.4.3 requests mrcfile gemmi streamlit st_clickable_images".split())

if __name__ == "__main__":
    main()

