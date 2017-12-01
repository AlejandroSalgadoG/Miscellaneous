public class Employee extends Worker{

    public Employee(String name){
        super(name);
    }

    public void displayStructure(){
        System.out.println(name);
    }

    public void add(Worker worker){}
    public void remove(Worker worker){}

}
