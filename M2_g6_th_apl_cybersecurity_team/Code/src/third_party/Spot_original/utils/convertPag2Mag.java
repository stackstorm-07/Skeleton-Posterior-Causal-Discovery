// package utils;

import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import java.io.File;

public class convertPag2Mag {

    private static void convert(String path){
        File graph = new File(path);

        if (!graph.getName().endsWith(".pag")){
            System.out.println(graph.getName().concat(" --- DOES NOT EXIST"));
            return;
        }

        try {
            Graph pag = GraphUtils.loadGraphTxt(graph);
            Graph mag = SearchGraphUtils.pagToMag(pag);
            File out = new File(graph.getParent().concat("/").concat(graph.getName()).concat(".mag"));
            GraphUtils.saveGraph(mag, out, false);
        }

        catch(Exception e){
            System.out.println(graph.getName().concat(" --- ERROR"));
        }
    }

    public static void main(String[] args) {
        convert(args[0]);
    }
}