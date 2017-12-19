# Four-Point-Invoice-Transform-with-OpenCV

This code is inspired from <a hred="4 Point OpenCV getPerspective Transform Example">[4 Point OpenCV getPerspective Transform Example]</a>

I have customized the code of <a href="https://twitter.com/PyImageSearch">Adrian</a> to find <b>4 points</b> of document or rectangle dynamically. Here i have added <I>findLargestCountours</I> and <I>convert_object</I>, where convert_object is our driver method which actually doing image processing and getting all 4 point rectangles from image. After getting all 4 point rectangle list <I>findLargestCountours<I> method finding  largest countour in list.

Here are some examples.

<Table>
    <tr>
        <th>Original Image</th>
        <th>Edge Detection</th>
        <th>Warped Image</th>
    </tr>
    <tr>
        <td><img src="https://raw.githubusercontent.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/master/Sample2/Original.png" alt="Original" width="400" height="500" align="middle"/></td>
        <td><img src="https://raw.githubusercontent.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/master/Sample2/%20Screen.png" alt="Screen" width="400" height="500" align="middle"/></td>
        <td><img src="https://raw.githubusercontent.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/master/Sample2/warp.png" alt="Warped" width="400" height="500" align="middle"/></td>
    </tr>
     <tr>
        <td><img src="https://raw.githubusercontent.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/master/Sample3/Original.png" alt="Original" width="400" height="500" align="middle"/></td>
        <td><img src="https://raw.githubusercontent.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/master/Sample3/%20Screen.png" alt="Screen" width="400" height="500" align="middle"/></td>
        <td><img src="https://raw.githubusercontent.com/KMKnation/Four-Point-Invoice-Transform-with-OpenCV/master/Sample3/warp.png" alt="Warped" width="400" height="500" align="middle"/></td>
    </tr>
</Table>


