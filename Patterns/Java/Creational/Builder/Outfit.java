import java.util.ArrayList;
import java.util.List;

public class Outfit{

    private List<Clothe> shoped;

    Outfit(){
        shoped = new ArrayList<Clothe>();
    }

    public void addClothe(Clothe clothe){
        shoped.add(clothe);
    }

    public void listClothe(){
        System.out.println("Outfit: ");
        for(Clothe clothe: shoped)
            System.out.println( "\t" + clothe.getName() );
    }

}
