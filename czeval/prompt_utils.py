import string


def load_prompt_file(prompt_file):
    with open(prompt_file, "r") as f:
        prompt = f.read()
    return prompt


class DotAwareFormatter(string.Formatter):
    """
    Allows us to use a.b in format keys, e.g. {user.name}.
    """

    def get_field(self, field_name, args, kwargs):
        return (self.get_value(field_name, args, kwargs), field_name)


def prepare_conversations(sample, system_prompt, user_prompt, user_variables):
    conversations = []
    conversations.append({"role": "system", "content": system_prompt})
    formatter = DotAwareFormatter()

    conversations.append(
        {
            "role": "user",
            "content": formatter.vformat(
                user_prompt, (), {key: sample[key] for key in user_variables}
            ),
        }
    )
    return conversations
