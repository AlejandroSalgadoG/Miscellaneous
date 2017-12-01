import java.util.List;
import java.util.ArrayList;

public class Boss extends Worker{

    private List<Worker> workers;

    public Boss(String name){
        super(name);
        workers = new ArrayList<Worker>();
    }

    public void add(Worker worker){
        workers.add(worker);
    }

    public void remove(Worker worker){
        workers.remove(worker);
    }

    public void displayStructure(){
        System.out.println(name);

        for(Worker worker : workers)
            worker.displayStructure();
    }

}
