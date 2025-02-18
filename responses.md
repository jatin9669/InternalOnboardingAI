explain about high level architecture of the team

The high-level architecture of the team primarily revolves around a monolithic application, referred to as the RCE Backend, which is built using Ruby on Rails (RoR). Here are the key components and their interactions within the architecture:

EKS k8s Cluster: The core services operate within an Elastic Kubernetes Service (EKS) Kubernetes cluster.

RCE Backend (Monolith):

This is the central component that provides both internal and external APIs, including the Event Dashboard for event organizers.
It handles attendee authentication and stores data in Aurora PostgreSQL.
Caching is managed through RedisLabs, while job scheduling is handled by Sidekiq Enterprise.
It produces messages related to attendee activity that are sent to the AWS MSK Kafka bus for further processing.
Analytics:

Another Ruby on Rails application that consumes messages from the Kafka bus, aggregates data points from attendee web and mobile apps, and stores this data in a separate Aurora PostgreSQL database.
It tracks attendee engagement metrics, generates reports on demand, and provides an API to render analytics insights within the Event Dashboard.
Registrations:

A React web application that allows event organizers to configure ticketing options and presents a registration widget for attendees.
App Store:

A React web application enabling organizers to install and configure third-party applications like polls, audio translations, etc.
Page Builder (Canvas):

A React web application that allows organizers to create custom event landing pages using a rich-text editor.
Captions API:

Provides client-side credentials for the Azure translations service. Plans exist to merge this API into the monolith due to its small size.
Longwave:

This service calls webhooks to provide real-time attendee activity updates to event organizers.
Onsite:

Specific details on this component are incomplete in the provided context.
All data-related operations are handled within the database hosted on Aurora RDS PostgreSQL in the eu-west-1 region, automating replication between writer and reader replicas for high availability.

In summary, the high-level architecture emphasizes a cohesive system where various components interact to support event management, attendee engagement, and analytics, with a focus on ensuring data integrity and real-time responsiveness for users.

give the schema of existing user tables

The existing user tables in the RCE database and their descriptions are as follows:

users

Description: Base users table, stores various attributes related to users.
Key Columns:
id: Unique identifier for the user.
personal data: Stores personal attributes of users.
authentication data: Contains information related to user authentication.
payment attributes: Stores payment-related information.
settings attributes: Holds user settings.
admin_users

Description: Admin users table, which stores information specific to admin users.
Key Columns:
id: Unique identifier for the admin user.
authentication data: Contains authentication-related information for admins.
organisers

Description: Organization table, stores organization-related data.
Key Columns:
id: Unique identifier for the organization.
Other organization-specific data attributes (not specified in the provided context).
organization_members

Description: Link table connecting users to an organization and stating their role within that organization.
Key Columns:
user_id: References the unique identifier in the users table.
role: Indicates the role of the user within the organization. Possible roles include:
organization_admin
regular_member
event_admin
This schema outlines the existing structure related to users and their associations with organizations within the RCE system. The specific attributes for the users and organisers tables may contain additional details depending on implementation but are not explicitly defined in the provided context.

how are we using redis

Redis is utilized in the RCE architecture primarily for caching and improving performance in data retrieval operations. Here are the key uses of Redis:

Caching:

Redis is employed as a caching layer to store frequently accessed data temporarily. This helps reduce the load on the PostgreSQL database and speeds up data retrieval, especially for repeated requests for the same information.
Session Management:

Redis can be used to store user session data, allowing for quick access and updates to active sessions. This is particularly useful in scenarios where frequent session validation is necessary without constantly querying the primary database.
Job Queueing:

Redis works in conjunction with Sidekiq for job scheduling and background processing. It manages background jobs efficiently by utilizing queues, allowing the application to handle asynchronous tasks such as sending emails or processing user data outside of the main application flow.
Data Structures:

Redis supports various data structures (such as strings, lists, sets, etc.) that can facilitate different use cases, like maintaining counts (e.g., user engagement metrics) or managing temporary data that doesn't need to be permanently stored.
By leveraging Redis as a caching mechanism and for job processing, the RCE architecture enhances performance, scalability, and user experience while efficiently managing resources.

--------------------------------

what are events auth domain

The event authentication domain consists of four primary authentication "domains" relevant to events:

Events Account Domain: This domain is associated with the account used to create an organization. Event attendees also receive an events account authentication. This authentication is cookie-based and is maintained using a Rails extension called devise. It's primarily a website login system (referred to as "Rails Login") rather than an API integration.

Event Experience Domain: When entering an event, you request an event experience JWT token. This token is used to authenticate the event experience frontend to the event APIs. The JWT token is maintained for the duration of the event, and any attendee-related activities should primarily use this token. The Rails login is mainly for organization and event management, including obtaining the JWT for entering events.

External API Domain: Organizers who wish to use the external API for automation or scripting can obtain an External API token through customer support. This token has a long lifespan and is valid only for specific external API endpoints provided — it cannot be used for other endpoints.

Analytics Domain: A separate analytics project contains organizer analytics data. The organization dashboard uses a JWT generated through the Rails login to authenticate to the analytics server. This token is used solely for this purpose and the number of associated API endpoints is very limited.

In summary, these domains ensure that authentication is handled appropriately based on different aspects of event management and participation.

whats the other name for canvas and explain about it

The other name for "Canvas" is Page Builder.

Page Builder (Canvas): This is a React web application that allows event organizers to create custom landing pages for their events using a rich-text editor. The Page Builder enables organizers to design these landing pages before the event registration begins. With this tool, they can customize the layout, content, and appearance of the event's landing page, promoting the event and providing essential information to potential attendees. The landing page serves as a primary entry point for attendees to learn about the event and proceed to register.

Overall, the Page Builder plays a crucial role in enhancing the attendee experience by allowing for personalization and tailored communication regarding an event.

what is mar and mts

In the context of the information provided, MAR and MTS likely refer to two types of services or functionalities related to event management or participant tracking, though they were not explicitly defined in the provided documentation.

If MAR refers to Messaging and Reporting or a similar function, it could involve features related to communication and reporting metrics associated with event attendance or interactions.

MTS may refer to Mobile Tracking Service or a similar concept relating to tracking attendee engagement or activities through mobile applications.

However, without specific definitions or details provided in the context, this is based on common interpretations of such acronyms. If more precise information is needed, it's recommended to reach out to the relevant contacts mentioned for clarification.

explain about component and flows

In the context of the provided documentation, components and flows refer to the different interconnected services and their interactions within the EKS Kubernetes cluster used for managing events.

Components:
Monolith (Ruby on Rails application):

Provides both internal and external APIs.
Contains the Event Dashboard used by event organizers.
Handles authentication of attendees.
Data storage through Aurora PostgreSQL.
Uses RedisLabs for caching.
Manages job scheduling via Sidekiq Enterprise.
Produces messages related to attendee activity to the AWS MSK Kafka bus.
Analytics (Ruby on Rails application):

Consumes messages from the Kafka bus and data from attendee web/mobile apps.
Aggregates data into a separate database (also Aurora PostgreSQL).
Tracks time spent by attendees at various event locations.
Generates CSV reports and provides a data API to render metrics/charts in the Event Dashboard.
Registrations (React web application):

Allows event organizers to configure ticketing for events.
Renders a registration widget for attendee registration.
App Store (React web application):

Enables event organizers to install and configure third-party applications for their events (like polls, translations, etc.).
Page Builder (Canvas):

Facilitates event organizers in creating custom event landing pages using a rich-text editor, displayed before event registration.
Captions (API):

Provides client-side credentials to use the Azure translation service. It will be merged into the Monolith due to its smaller size.
Longwave:

Calls webhooks to provide real-time attendee activity updates to event organizers (such as registration notifications).
Onsite:

This component wasn't detailed in the documentation but could relate to on-site systems managing attendee interactions at the event.
Flows:
These refer to the sequences and processes that users (event organizers and attendees) go through while interacting with components.

Creating an Event:

An event organizer signs in and creates a new event through the Organiser dashboard.
Configures event details (dates, description, schedule, speakers, etc.) via the Event dashboard.
Optionally configures registration through the Registration dashboard.
Optionally uses the Page Builder to design a custom landing page.
Previews the event before it goes live.
Registering for an Event:

Participants visit the event landing page to register (e.g., purchasing tickets).
Organizers may invite participants to register for specific ticket types via email.
Data Flow:

Data is stored in the Aurora PostgreSQL database in the eu-west-1 cluster.
Populated through various components including the Monolith backend server and the Analytics API.
Replication of data occurs automatically between writer and reader replicas in the database setup.
These flows and components demonstrate how various parts of the system interact to offer a seamless experience for both event organizers and attendees, ensuring efficient event management and engagement.