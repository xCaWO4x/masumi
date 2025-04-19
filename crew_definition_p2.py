from crewai import Agent, Crew, Task
from composio_crewai import ComposioToolSet, App


toolset = ComposioToolSet(api_key="a0lvtzodkm6rhnj17ezath")
tools = toolset.get_tools(apps=[App.GMAIL])

# check list of email tools that are available
#for tool in tools:
#    print(tool.name)

class EmailDispatchCrew:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.crew = self.create_crew()

    def create_crew(self):
        # Agent 1: Extract structured JSON from natural language
        drafter = Agent(
            role='Email Info Extractor',
            goal='Extract recipients and email context from natural language input and format as clean JSON',
            backstory='Skilled in interpreting unstructured text and producing structured outreach targets and context.',
            verbose=self.verbose
        )

        # Agent 2: Format email drafts for Gmail API
        formatter = Agent(
            role='Email Formatter',
            goal='Generate email drafts from input context and prepare full GMAIL_SEND_EMAIL-compatible parameters',
            backstory='Expert in message construction and parameter mapping for automation pipelines.',
            verbose=self.verbose
        )

        # Agent 3: Send each email upon user confirmation
        dispatcher = Agent(
            role='Email Sender',
            goal='Send emails using Gmail API, but only after user confirms each one',
            backstory='Trusted with controlled, confirmable delivery of outbound emails through API integration.',
            tools=tools,
            verbose=self.verbose
        )

        crew = Crew(
            agents=[drafter, formatter, dispatcher],
            tasks=[
                Task(
                    description=(
                        "Parse ONLY the provided input text to extract:\n"
                        "- A list of recipients (DO NOT invent): each with name, email, firm\n"
                        "- An email_context object with: goal, startup_name, value_prop, traction, ask, sender_name, sender_role\n\n"
                        "Output strictly valid JSON in this format:\n"
                        "{\n"
                        "  \"recipients\": [\n"
                        "    {\"name\": \"Howard Chao\", \"email\": \"jchao1@wpi.edu\", \"firm\": \"Sequoia Capital\"},\n"
                        "    {\"name\": \"Jhih-ci Chao\", \"email\": \"cawo4edx@gmail.com\", \"firm\": \"a16z\"}\n"
                        "  ],\n"
                        "  \"email_context\": {\n"
                        "    \"goal\": \"Raise a $5M seed round\",\n"
                        "    \"startup_name\": \"Neuroflux\",\n"
                        "    \"value_prop\": \"N/A\",\n"
                        "    \"traction\": \"$200K ARR and 3 pilots\",\n"
                        "    \"ask\": \"Book intro calls\",\n"
                        "    \"sender_name\": \"John Doe\",\n"
                        "    \"sender_role\": \"Founder & CEO\"\n"
                        "  }\n"
                        "}\n\n"
                        "Return ONLY this JSON. Do not make up any names or firms. Parse directly from input."
                    ),
                    expected_output="JSON object with keys: recipients[], email_context{}",
                    agent=drafter
                ),
                Task(
                    description=(
                        "For each recipient in the list, generate a personalized subject and body using the context. "
                        "Then format it into a full Gmail API send object with:\n"
                        "- user_id: 'me'\n"
                        "- recipient_email, subject, body\n"
                        "- optional: cc, bcc, extra_recipients, is_html (true), attachment (null)\n"
                        "Output a list of full email objects, one per recipient."
                    ),
                    expected_output='List of GMAIL_SEND_EMAIL-compatible objects',
                    agent=formatter,
                    input='result_of_previous_task'
                ),
                Task(
                    description=(
                        "Take the list of email objects. Ask the user for confirmation before sending each email. "
                        "If confirmed, call GMAIL_SEND_EMAIL with the object's data. Return status and response for each."
                    ),
                    expected_output='List of {recipient_email, status, response_data} objects',
                    agent=dispatcher,
                    input='result_of_previous_task'
                )
            ]
        )

        return crew
