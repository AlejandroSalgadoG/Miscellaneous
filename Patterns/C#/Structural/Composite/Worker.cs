public abstract class Worker{

    protected string name;

    public Worker(string name){
        this.name = name;
    }

    abstract public void add(Worker worker);
    abstract public void remove(Worker worker);
    abstract public void displayStructure();
}
