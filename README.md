# RecSys\_and\_LLM

### 0. 프로젝트 설정 가이드

`RecSys_and_LLM` 프로젝트를 설정하고 실행하려면, 아래 지침을 따르시면 됩니다.

### 1. Poetry 설치

Poetry는 Python 패키지 관리를 위한 도구입니다. 공식적으로 `pipx`를 통한 설치가 권장됩니다. 

#### pipx를 이용한 설치 (권장):

  ```bash
  pip install --user pipx 
  pipx install poetry
  ```

위 방법이 안될 시에는 운영체제에 따라 아래와 같은 대체 방법을 사용해보세요!

- **macOS / Linux**:

  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```

- **Windows (PowerShell)**:

  ```powershell
  (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
  ```


### 2. 환경 변수 설정
설치 후, 환경 변수를 설정해야 합니다. 설치 경로를 확인하고 해당 경로를 시스템의 `PATH`에 추가하세요. 

#### Windows:
1. **설치 경로 확인**: 설치 완료 후 출력된 경로를 확인합니다. ex: `C:\Users\USERNAME\AppData\Roaming\Python\Scripts`
2. 환경 변수 설정:
   - **시스템 환경 변수 편집**: `제어판` > `시스템 및 보안` > `시스템` > `고급 시스템 설정` > `환경 변수`
   - **시스템 변수**에서 `Path` 선택 후 `편집`
   - `새로 만들기`를 클릭하고 위에서 확인한 경로를 추가
3. 변경 사항 적용을 위해 **시스템 재시작**

#### Linux:
1. 터미널에서 `nano` 또는 다른 텍스트 편집기를 사용하여 `~/.bashrc` 파일을 엽니다:
```bash
nano ~/.bashrc
```
2. 줄 추가: 파일의 끝에 다음 줄을 추가합니다.  
```bash
export PATH="$HOME/.local/bin:$PATH"
```
3. 파일 저장 및 종료: nano에서는 `Ctrl + O`를 눌러 저장하고, `Ctrl + X`를 눌러 종료합니다.
4. 변경 사항 적용: 터미널에서 다음 명령어를 실행하여 변경 사항을 적용합니다.
```bash
source ~/.bashrc
```

### 3. 설치 확인
설치가 완료되면 다음 명령어로 확인하세요:

```bash
poetry --version
```

### 4. 저장소 클론

프로젝트를 로컬 머신에 클론합니다:

```bash
git clone https://github.com/PNUDI/RecSys_and_LLM.git
cd RecSys_and_LLM
```

### 5. 가상 환경 설정

Poetry를 사용하여 의존성을 설치하고 가상 환경을 설정합니다:

```bash
poetry install
```

이 명령어는 `pyproject.toml`과 `poetry.lock`에 명시된 모든 의존성을 설치합니다. 

#### 연구실 서버에서 poetry install 시 Pending 이슈 발생 시 
keyring 관련 문제라면
아래 명령어를 bash 터미널에 입력하면 해결이 가능합니다.  

```bash
poetry config keyring.enabled false
```

다시 poetry install 해주시면 됩니다. 
자세한 사항은 [서버 poetry 설치 무한 pending 이슈 해결방법](https://velog.io/@diluny/%EC%84%9C%EB%B2%84-poetry-%EC%84%A4%EC%B9%98-%EB%AC%B4%ED%95%9C-pending-%EC%9D%B4%EC%8A%88-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95) 참고


**❗이제부터 모든 명령은 poetry run ~~ 와 같은 형식으로 진행됩니다.**



### 6. Pre-commit 훅 설정

코드 스타일과 포맷팅을 유지하기 위해 `pre-commit`을 설정합니다:

1. `pre-commit` 설치:

   ```bash
   poetry run pre-commit install
   ```

2. 훅 업데이트:

   ```bash
   poetry run pre-commit autoupdate
   ```

3. 모든 파일에 대해 수동으로 훅 실행:

   ```bash
   poetry run pre-commit run --all-files
   ```



### 7. 프로젝트 실행

**🥳 축하합니다!**  
이제 가상 환경이 활성화된 상태에서 **다음과 같이 프로젝트를 실행**할 수 있습니다:

```bash
poetry run streamlit run recsys_and_llm\front\app.py
```

---

### A. 문제 해결 🛠️

- **`pre-commit`**** 훅 또는 환경 관련 문제 발생 시**:

  ```bash
  poetry run pre-commit clean
  ```

- **의존성 설치 실패 시**: `pyproject.toml`에 명시된 Python 버전(`>=3.10.0`)과 현재 사용 중인 Python 버전이 일치하는지 확인하세요.
- 2025.02.10(월) 기준 사용하기로 약속한 Python 버전은 3.10.12 입니다.

---

### B. 참고 사항 ⚠️

- 모든 의존성은 `poetry`를 통해 관리됩니다. **`pip`을 사용하여 패키지를 직접 설치하지 마세요.**
- 원활한 설정을 위해 **`poetry`가 전역적으로 설치되어 있는지 확인**하세요.

