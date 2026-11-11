export class LoadForecast {
  constructor(repository) {
    this.repository = repository;
  }

  execute() {
    return this.repository.getCurrent();
  }
}
