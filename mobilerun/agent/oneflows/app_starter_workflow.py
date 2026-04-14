"""
Simple workflow to open an app based on a description.
"""

import json
import logging

from workflows import Context, Workflow, step
from workflows.events import StartEvent, StopEvent

from droidrun.agent.utils.inference import acomplete_with_retries

logger = logging.getLogger("droidrun")


class AppStarter(Workflow):
    """
    A simple workflow that opens an app based on a description.

    The workflow uses an LLM to intelligently match the app description
    to an installed app's package name, then opens it.
    """

    def __init__(self, driver, llm, timeout: int = 60, stream: bool = False, **kwargs):
        """
        Initialize the OpenAppWorkflow.

        Args:
            driver: DeviceDriver with get_apps()/start_app() methods
            llm: An LLM instance (e.g., OpenAI) to determine which app to open
            timeout: Workflow timeout in seconds (default: 60)
            stream: If True, stream LLM response to console in real-time
            **kwargs: Additional arguments passed to Workflow
        """
        super().__init__(timeout=timeout, **kwargs)
        self.driver = driver
        self.llm = llm
        self.stream = stream

    @step
    async def open_app_step(self, ev: StartEvent, ctx: Context) -> StopEvent:
        """
        Opens an app based on the provided description.

        Expected StartEvent attributes:
            - app_description (str): The name or description of the app to open

        Returns:
            StopEvent with the result of the open_app operation
        """
        app_description = ev.app_description

        # Get list of installed apps
        apps = await self.driver.get_apps(include_system=True)

        # Format apps list for LLM
        apps_list = "\n".join(
            [
                f"- {app['label']} (package: {app['package_name'] if 'package_name' in app else app['package']})"
                for app in apps
            ]
        )

        # Construct prompt for LLM
        prompt = f"""Given the following list of installed apps and a user's description, determine which app package name to open.

Installed Apps:
{apps_list}

User's Request: "{app_description}"

Return ONLY a JSON object with the following structure:
{{
    "package": "com.example.package"
}}

If no app matches the user's request, return:
{{
    "package": null
}}

Choose the most appropriate app based on the description. Return the package name of the best match, or null if no app matches."""

        # Get LLM response
        logger.info("📱 AppOpener response:", extra={"color": "blue"})
        response = await acomplete_with_retries(self.llm, prompt, stream=self.stream)
        response_text = response.text.strip()

        # Parse JSON response - extract content between { and }
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            result_json = json.loads(json_str)
            package_name = result_json["package"]
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return StopEvent(
                result=f"Error parsing LLM response: {e}. Response: {response_text}"
            )

        if not package_name:
            logger.warning(f"No matching app found for: {app_description}")
            return StopEvent(
                result=f"Could not open app: no installed app matches '{app_description}'"
            )

        logger.info(f"Starting app {package_name}")
        result = await self.driver.start_app(package_name)

        return StopEvent(result=result)


# Example usage
async def main():
    """
    Example of how to use the OpenAppWorkflow.
    """
    from llama_index.llms.openai import OpenAI

    from droidrun.tools.driver.android import AndroidDriver

    # Initialize driver with device serial (None for default device)
    driver = AndroidDriver(serial=None)
    await driver.connect()

    # Initialize LLM
    llm = OpenAI(model="gpt-4o-mini")

    # Create workflow instance
    workflow = AppStarter(driver=driver, llm=llm, timeout=60, verbose=True)

    # Run workflow to open an app
    result = await workflow.run(app_description="Settings")

    print(f"Result: {result}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
